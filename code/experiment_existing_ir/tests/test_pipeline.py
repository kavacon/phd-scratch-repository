import subprocess
import unittest

from compiler.parser.parser import parse
from experiment_existing_ir import pipeline
from experiment_existing_ir.translate import TranslationError, translate

ARITHMETIC = "x = 5; y = x + 3; y = y - 1; return y;"
TIED = (*pipeline.PREPARE, *pipeline.RESOLVE, "--tie-instances", "--canonicalize", "--lower-to-qc")
CONDITIONAL = "a = 0..4; x = 2; if a < 2 { x += 3; } else { x -= 1; } return x;"


def program(source: str, grammar: str = "v1", **options) -> str:
    return translate(parse(source, grammar), **options)


def optimise(source: str, *passes: str) -> str:
    """The program after its loops are unrolled, its slots promoted and its references resolved, then the given passes."""
    return pipeline.optimise(source, *pipeline.PREPARE, *pipeline.RESOLVE, *passes)


def registers(qc: str) -> int:
    return qc.count("memref.alloc")


@unittest.skipUnless(pipeline.built(), "existing-ir-opt is not built: run helper/cpp/setup.sh and cpp/build.sh")
class Pipeline(unittest.TestCase):
    def test_a_definite_program_returns_its_value_with_and_without_folding(self):
        for passes in (("--lower-to-qc",), ("--canonicalize", "--lower-to-qc")):
            self.assertEqual(pipeline.run(program(ARITHMETIC), (*pipeline.PREPARE, *passes)), {7: 2000}, passes)

    def test_folding_the_source_first_removes_the_arithmetic_from_the_circuit(self):
        plain = pipeline.optimise(program(ARITHMETIC), *pipeline.PREPARE, "--lower-to-qc")
        folded = pipeline.optimise(program(ARITHMETIC), *pipeline.PREPARE, "--canonicalize", "--lower-to-qc")
        # the constants 5, 3 and 1 are written into the registers that use them, so they are none of the six: the registers of x and y,
        # and for each of the two sums an operand temporary and a carry
        self.assertEqual((registers(plain), "qc.call" in plain), (6, True))
        self.assertEqual((registers(folded), "qc.call" in folded), (1, False))  # 7 is 0111, in one register

    def test_arithmetic_wraps_modulo_the_type_width(self):
        # 7 + 7 is 14, which is -2 in four bits of two's complement
        self.assertEqual(pipeline.run(program("a = 7; b = a + a; return b;", width=4), shots=100), {-2: 100})

    def test_a_conditional_on_a_distribution_gives_each_outcome_with_its_probability(self):
        # a is 0, 1, 2 or 3 with equal probability, so x is 5 when a < 2 and 1 otherwise: each half the time
        counts = pipeline.run(program(CONDITIONAL, "v2"))
        self.assertEqual(set(counts), {1, 5})
        self.assertTrue(all(800 < count < 1200 for count in counts.values()), counts)

    def test_an_in_place_update_shifts_every_outcome(self):
        # a is 0, 1, 2 or 3, each a quarter of the time; the first result fits a's register, the second needs a wider one
        for source, outcomes in (("a = 0..4; a -= 1; return a;", {-1, 0, 1, 2}), ("a = 0..4; a += 3; return a;", {3, 4, 5, 6})):
            counts = pipeline.run(program(source, "v2"))
            self.assertEqual(set(counts), outcomes, source)
            self.assertTrue(all(400 < count < 600 for count in counts.values()), (source, counts))

    def test_updates_reuse_the_distribution_s_register_however_far_they_widen_it(self):
        # the distribution, and for each update only a temporary and a carry: no fresh result register, and the constants are written
        # into the temporaries and have no register of their own
        source = program("a = 0..4; a += 3; a += 5; a -= 20; return a;", "v2")
        lowered = optimise(source, "--canonicalize", "--in-place-updates", "--lower-to-qc")
        self.assertEqual(registers(lowered), 1 + 3 * 2)
        self.assertEqual(set(pipeline.run(source)), {-12, -11, -10, -9})  # 0..3 + 3 + 5 - 20

    def test_adding_a_distribution_in_place_gives_the_distribution_of_the_sum(self):
        # a is 0..3 and p is 0..1 independently, so a + p is 0 to 4 with weights 1, 2, 2, 2, 1 out of 8
        counts = pipeline.run(program("a = 0..4; p = 0..2; a += p; return a;", "v2"), shots=4000)
        self.assertEqual(set(counts), {0, 1, 2, 3, 4})
        for value, weight in zip(range(5), (1, 2, 2, 2, 1)):
            self.assertAlmostEqual(counts[value] / 4000, weight / 8, delta=0.04, msg=value)

    def test_a_name_is_an_alias_and_one_expression_uses_one_roll(self):
        shift = pipeline.run(program("a = 0..4; b = a; b += 1; return a;", "v2"))  # the update is seen through both names
        self.assertEqual(set(shift), {1, 2, 3, 4})
        self.assertEqual(pipeline.run(program("a = 0..4; return a - a;", "v2"), shots=200), {0: 200})
        self.assertEqual(set(pipeline.run(program("a = 0..4; return a + a;", "v2"))), {0, 2, 4, 6})

    def test_a_computed_assignment_is_independent_of_its_inputs(self):
        # c = a + 1 is an independent die, so c - a spreads from -2 to 4 with triangular weights out of 16
        counts = pipeline.run(program("a = 0..4; c = a + 1; return c - a;", "v2"), shots=4000)
        self.assertEqual(set(counts), set(range(-2, 5)))
        for value, weight in zip(range(-2, 5), (1, 2, 3, 4, 3, 2, 1)):
            self.assertAlmostEqual(counts[value] / 4000, weight / 16, delta=0.04, msg=value)

    def test_extracting_a_temporary_changes_the_result_when_the_variable_is_used_again(self):
        one_expression = pipeline.run(program("a = 0..4; c = (a + 1) + a; return c;", "v2"))
        self.assertEqual(set(one_expression), {1, 3, 5, 7})  # one roll used twice: 2a + 1
        extracted = pipeline.run(program("a = 0..4; t = a + 1; c = t + a; return c;", "v2"), shots=4000)
        self.assertEqual(set(extracted), set(range(1, 8)))  # t is independent of a: a sum of two dice

    def test_tying_instances_is_the_alternative_meaning_and_makes_the_difference_always_one(self):
        self.assertEqual(pipeline.run(program("a = 0..4; c = a + 1; return c - a;", "v2"), TIED, shots=200), {1: 200})

    def test_instances_must_be_resolved_before_lowering(self):
        with self.assertRaises(subprocess.CalledProcessError):
            optimise(program("a = 0..4; c = a + 1; return c;", "v2"), "--lower-to-qc")

    def test_replaying_a_history_duplicates_its_distributions_and_tying_does_not(self):
        source = program("a = 0..4; c = a + 1; return c - a;", "v2")
        replayed = optimise(source, "--resolve-instances", "--canonicalize")
        tied = optimise(source, "--tie-instances", "--canonicalize")
        self.assertEqual((replayed.count("dist.uniform"), tied.count("dist.uniform")), (2, 1))

    def test_the_standard_cse_pass_does_not_merge_independent_distributions(self):
        # two identical ranges are two dice, and two instances of one value are two draws
        cse = optimise(program("a = 0..4; c = a + 1; d = a + 1; return c - d;", "v2"), "--cse")
        self.assertEqual(cse.count("dist.instance"), 2)
        replayed = optimise(program("a = 0..4; c = a + 1; return c - a;", "v2"), "--resolve-instances", "--cse")
        self.assertEqual(replayed.count("dist.uniform"), 2)

    def test_a_sum_whose_first_operand_dies_becomes_an_in_place_update_and_keeps_its_meaning(self):
        base = ("--resolve-instances", "--canonicalize")
        source = program("a = 0..4; t = a + 1; c = t + a; return c;", "v2")
        converted = optimise(source, *base, "--in-place-updates")
        self.assertEqual((converted.count("dist.addi"), converted.count("dist.add_assign")), (0, 2))
        before = registers(optimise(source, *base, "--lower-to-qc"))
        after = registers(optimise(source, *base, "--in-place-updates", "--lower-to-qc"))
        self.assertLess(after, before)  # the result registers are gone
        counts = pipeline.run(source, shots=4000)
        self.assertEqual(set(counts), set(range(1, 8)))  # the same sum of two independent dice

    def test_a_value_that_is_still_used_is_not_consumed_by_the_conversion(self):
        # in (a + 1) - a the range `a` is used twice, so only the outer difference, whose first operand dies, converts
        source = program("a = 0..4; return (a + 1) - a;", "v2")
        converted = optimise(source, "--canonicalize", "--in-place-updates")
        self.assertEqual((converted.count("dist.addi"), converted.count("dist.sub_assign")), (1, 1))
        self.assertEqual(pipeline.run(source, shots=200), {1: 200})  # still always 1: the register of `a` is untouched
        self.assertEqual(optimise(program("a = 0..4; return a - a;", "v2"),
                                           "--in-place-updates").count("dist.sub_assign"), 0)

    def test_an_instance_starts_a_group_of_its_own_until_it_is_tied(self):
        source = program("a = 0..4; c = a + 1; return c;", "v2")
        self.assertIn("dist.groups = 2 : i64", optimise(source, "--dist-status"))
        self.assertIn("dist.groups = 1 : i64", optimise(source, "--tie-instances", "--dist-status"))

    def test_a_loop_is_unrolled_into_straight_line_code_that_gives_the_sum_of_its_iterations(self):
        # 0 + 1 + 2 + 3, then a nest (i + j over 3 x 3 is 18) and a loop that runs no times; each is folded or lowered whole
        for source, value in (("s = 0; for i in 0..4 { s += i; } return s;", 6),
                              ("s = 0; for i in 0..3 { for j in 0..3 { t = i + j; s += t; } } return s;", 18),
                              ("s = 5; for i in 2..2 { s += 1; } return s;", 5),
                              ("a = 1; b = 10; for i in 0..3 { a += i; b -= a; } return b;", 10 - 1 - 2 - 4)):
            self.assertEqual(pipeline.run(program(source), shots=100), {value: 100}, source)
        unrolled = optimise(program("s = 0; for i in 0..4 { s += i; } return s;"), pipeline.UNROLL)
        self.assertNotIn("affine.for", unrolled)

    def test_an_integer_becomes_a_distribution_when_a_distribution_is_added_to_or_assigned_to_it(self):
        # += keeps x tied to a, so the difference is always 2; assigning a computed value is an independent instance
        self.assertEqual(pipeline.run(program("a = 0..4; x = 2; x += a; return x - a;", "v2"), shots=200), {2: 200})
        self.assertEqual(pipeline.run(program("a = 0..4; x = 2; x -= a; return x + a;", "v2"), shots=200), {2: 200})
        counts = pipeline.run(program("a = 0..4; x = 2; x = a + 1; return x - a;", "v2"), shots=4000)
        self.assertEqual(set(counts), set(range(-2, 5)))

    def test_an_integer_that_becomes_a_distribution_on_some_path_through_a_branch_or_loop(self):
        # (source, outcomes, hand-worked weights); a is 0..3, so 2 + a is 2..5 and a sum of three copies of a is x - 2
        for source, expected in (
                ("a = 0..4; x = 2; k = 5; if k < 3 { x += a; } y = x + 1; return y;", {3}),  # the branch is dead: x stays 2
                ("a = 0..4; x = 2; k = 1; if k < 3 { x += a; } return x;", {2, 3, 4, 5}),
                ("a = 0..4; x = 2; for i in 0..3 { x += a; } return x - a - a - a;", {2}),  # x is tied to a on every pass
                ("a = 0..4; x = 0; for i in 0..6 { if i < 3 { x += a; } else { x -= 1; } } return x - a - a - a;", {-3}),
                ("a = 0..4; c = 0; for i in 0..3 { for j in 0..3 { if i < j { c += a; } } } return c - a - a - a;", {0})):
            self.assertEqual(set(pipeline.run(program(source, "v2"), shots=500)), expected, source)
        # three independent one-bit dice made inside a loop: 0 to 3 with weights 1, 3, 3, 1 out of 8
        counts = pipeline.run(program("s = 0; for i in 0..3 { d = 0..2; s += d; } return s;", "v2"), shots=4000)
        for value, weight in zip(range(4), (1, 3, 3, 1)):
            self.assertAlmostEqual(counts[value] / 4000, weight / 8, delta=0.04, msg=value)
        # an arm of a conditional on a distribution may do it too: 2 at 5/8 and 3, 4, 5 at 1/8 each
        counts = pipeline.run(program("a = 0..4; b = 0..2; x = 2; if b < 1 { x += a; } return x;", "v2"), shots=4000)
        for value, weight in zip(range(2, 6), (5, 1, 1, 1)):
            self.assertAlmostEqual(counts[value] / 4000, weight / 8, delta=0.04, msg=value)

    def test_a_condition_on_a_variable_that_a_dead_branch_could_superpose_is_decided_by_folding(self):
        # the branch is dead, so x stays 2 and the condition on it is taken every time
        source = "a = 0..4; x = 2; y = 0; k = 9; for i in 0..3 { if k < 3 { x += a; } if x < 5 { y += 1; } } return y;"
        self.assertEqual(pipeline.run(program(source, "v2"), shots=100), {3: 100})

    def test_conditionals_and_loops_on_definite_values_nest_in_either_order(self):
        # i < 2 adds 10 twice and the rest add 1 twice; a nest counts the pairs i < j; a local reset each pass; a loop in an arm
        for source, value in (("s = 0; for i in 0..4 { if i < 2 { s += 10; } else { s += 1; } } return s;", 22),
                              ("s = 0; for i in 0..3 { for j in 0..3 { if i < j { s += 1; } } } return s;", 3),
                              ("s = 0; for i in 0..4 { t = 1; if i < 2 { t = 5; } s += t; } return s;", 12),
                              ("s = 0; x = 1; if x < 3 { for i in 0..4 { s += i; } } else { s += 100; } return s;", 6)):
            self.assertEqual(pipeline.run(program(source), shots=100), {value: 100}, source)

    def test_a_call_copies_a_definite_argument_and_shares_a_distribution_argument(self):
        # a definite argument is a copy, so the caller's x stays 2 and the result is 2 + 5
        self.assertEqual(pipeline.run(program("function f(p) { p += 3; return p; }\nx = 2; y = f(x); return x + y;", "v2"),
                                      shots=100), {7: 100})
        # a distribution is passed as itself, so the callee's update is the caller's, and a result that is the parameter
        # is the same object: a is 0..3, bump makes it 1..4, and b += 10 makes it 11..14
        bump = "function bump(p) { p += 1; return p; }\n"
        self.assertEqual(set(pipeline.run(program(bump + "a = 0..4; b = bump(a); return a;", "v2"))), {1, 2, 3, 4})
        self.assertEqual(set(pipeline.run(program(bump + "a = 0..4; b = bump(a); b += 10; return a;", "v2"))),
                         {11, 12, 13, 14})
        # a result that is computed (p + 1) is a new instance, independent of its argument, so c - a spreads from -2 to 4
        plus = "function plus(p) { return p + 1; }\n"
        self.assertEqual(set(pipeline.run(program(plus + "a = 0..4; c = plus(a); return c - a;", "v2"), shots=4000)),
                         set(range(-2, 5)))

    def test_a_conditional_on_a_definite_value_lowers_only_once_it_has_become_a_select(self):
        source = "x = 2; if x < 3 { x += 5; } else { x -= 1; } return x;"
        self.assertEqual(pipeline.run(program(source)), {7: 2000})  # canonicalize folds the arms and the select
        with self.assertRaises(subprocess.CalledProcessError):  # a condition held in qubits cannot be branched on
            pipeline.optimise(program(source), *pipeline.PREPARE, "--lower-to-qc")

    def test_conditionals_on_a_distribution_nest_and_may_update_a_distribution(self):
        # a < 2 half the time and b < 1 half the time, independently, in each of two passes: s is 2 a quarter of the time
        counts = pipeline.run(program("a = 0..4; b = 0..2; s = 0; for i in 0..2 { if a < 2 { if b < 1 { s += 1; } } } return s;",
                                      "v2"), shots=4000)
        self.assertEqual(set(counts), {0, 2})
        self.assertAlmostEqual(counts[0] / 4000, 0.75, delta=0.04)  # b is one roll: both passes add or neither does
        self.assertAlmostEqual(counts[2] / 4000, 0.25, delta=0.04)
        # the loop of a < 2 (R10): all three passes add, or none does
        counts = pipeline.run(program("a = 0..4; s = 0; for i in 0..3 { if a < 2 { s += 1; } } return s;", "v2"), shots=4000)
        self.assertEqual(set(counts), {0, 3})
        # an update of a distribution under a condition, also through a call: a shifts by one when b < 1 (half the time)
        bump = "function bump(p) { p += 1; return p; }\n"
        for source in ("a = 0..4; b = 0..2; if b < 1 { a += 1; } return a;",
                       bump + "a = 0..4; b = 0..2; if b < 1 { bump(a); } return a;"):
            counts = pipeline.run(program(source, "v2"), shots=8000)
            for value, weight in zip(range(5), (1, 2, 2, 2, 1)):
                self.assertAlmostEqual(counts[value] / 8000, weight / 8, delta=0.03, msg=source)

    def test_a_distribution_added_to_itself_is_twice_one_roll(self):
        self.assertEqual(set(pipeline.run(program("a = 0..4; a += a; return a;", "v2"))), {0, 2, 4, 6})

    def test_a_name_aliased_inside_a_branch_is_the_distribution_only_when_the_branch_is_live(self):
        # x = a in a live branch makes x the object a, so x += 1 shifts a to 1..4; in a dead branch x stays the integer 2
        live = pipeline.run(program("a = 0..4; x = 2; k = 1; if k < 3 { x = a; } x += 1; return a;", "v2"))
        dead = pipeline.run(program("a = 0..4; x = 2; k = 9; if k < 3 { x = a; } x += 1; return a;", "v2"))
        self.assertEqual((set(live), set(dead)), ({1, 2, 3, 4}, {0, 1, 2, 3}))

    def test_a_comparison_kept_in_a_variable_is_a_condition_and_a_number_is_true_when_it_is_not_zero(self):
        self.assertEqual(pipeline.run(program("x = 3; c = x < 5; if c { x += 1; } return x;", "v2")), {4: 2000})
        self.assertEqual(pipeline.run(program("c = -2 < 1; return c;", "v2")), {1: 2000})
        counts = pipeline.run(program("a = 0..2; x = 0; if a { x = 7; } return x;", "v2"), shots=3000)  # a is 0 or 1
        self.assertEqual(set(counts), {0, 7})
        self.assertAlmostEqual(counts[0] / 3000, 1 / 2, delta=0.05)

    def test_what_the_semantics_refuse_is_refused_with_a_reason(self):
        # overwriting a distribution would drop it; a name bound to different objects in the arms of a conditional on a
        # distribution would mean an object that depends on the outcome
        for source, message in (("a = 0..4; a = 5; return a;", "would drop"),
                                ("a = 0..4; b = 0..2; y = a; x = 2; if b < 1 { x = y; } return x;", "different objects")):
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                pipeline.optimise(program(source, "v2"), *pipeline.DEFAULT)
            self.assertIn(message, caught.exception.stderr)


if __name__ == "__main__":
    unittest.main()

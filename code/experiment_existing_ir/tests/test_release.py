import math
import re
import subprocess
import unittest

from compiler.parser.parser import parse
from compiler.system import System
from experiment_existing_ir import metrics, pipeline
from experiment_existing_ir.translate import translate
from tests.golden import GOLDEN, TOLERANCE, distance, entry

# Programs that only fit a small system because dead registers are released and their qubits reused: the qubits each needs without
# release are more than the budget here. The intended outcome does not depend on the budget.
TIGHT = {"a function called in a loop": 20, "a loop variable compared with a distribution": 28,
         "a condition on a distribution in a loop": 19, "two independent conditional updates joined by a sum": 25,
         "three independent calls joined by sums": 29, "every comparison of a distribution": 27,
         "two variables tied to one condition": 29, "a distribution conditional inside another, in a loop": 24,
         "three independent comparisons joined by and": 21, "two independent accumulators in a loop joined by a sum": 24}
# These give the wrong outcome when a register is freed without being returned to zero first, because the qubits are reused (the last
# two of TIGHT happen not to reuse a dirty qubit in a way that shows).
SENSITIVE = tuple(TIGHT)[:8]


def compile_with(name: str, qubits=None, strategy="lazy", uncompute=True, policy="aware") -> str:
    program = translate(parse(entry(name).source, "v2"))
    if qubits is not None:
        program = pipeline.with_system(program, System(qubits=qubits))
    passes = tuple(p.replace("strategy=lazy", f"strategy={strategy}").replace("--release-registers", f"--release-registers{{uncompute={str(uncompute).lower()}}}")
                   .replace("--assign-qubits", f"--assign-qubits{{policy={policy}}}")
                   for p in pipeline.DEFAULT)
    return pipeline.optimise(program, *passes)


def pool(qc: str) -> int:
    return int(re.search(r"pool_qubits = (\d+)", qc).group(1))


def outcome(name: str, qc: str) -> float:
    return distance(pipeline.execute(pipeline.export(qc, "openqasm3"), "openqasm3", "aer", 3000), entry(name).outcomes)


@unittest.skipUnless(pipeline.built(), "existing-ir-opt is not built: run helper/cpp/setup.sh and cpp/build.sh")
class Release(unittest.TestCase):
    def test_a_program_fits_the_qubits_release_gives_it_and_still_gives_the_intended_outcome(self):
        for name, qubits in TIGHT.items():
            with self.subTest(name):
                self.assertGreater(pool(compile_with(name)), qubits)  # without release it needs more than the system has
                tight = compile_with(name, qubits)
                self.assertLessEqual(pool(tight), qubits)
                self.assertLess(outcome(name, tight), TOLERANCE)

    def test_a_program_that_does_not_fit_even_with_release_is_refused_and_a_system_that_is_large_enough_releases_nothing(self):
        name = "a loop variable compared with a distribution"
        with self.assertRaises(subprocess.CalledProcessError) as raised:
            compile_with(name, 10)
        self.assertIn("even releasing every register that may be released", raised.exception.stderr)
        self.assertIn("released_registers = 0", compile_with(name, 100))  # lazy: nothing needs releasing

    def test_releasing_eagerly_gives_the_intended_outcome_for_every_program(self):
        for golden in GOLDEN:
            with self.subTest(golden.name):
                self.assertLess(outcome(golden.name, compile_with(golden.name, strategy="eager")), TOLERANCE)

    def test_the_outcome_depends_on_returning_a_register_to_zero_before_its_qubits_are_reused(self):
        for name in SENSITIVE:
            with self.subTest(name):
                broken = compile_with(name, TIGHT[name], uncompute=False, policy="lowest")  # reuse as hard as possible
                self.assertGreater(outcome(name, broken), 0.2)  # the same qubits, reused dirty: not the intended outcome

    def test_given_more_qubits_than_it_needs_a_program_uses_them_to_wait_less_and_never_more_than_it_is_given(self):
        name = "a loop variable compared with a distribution"
        needed = pool(compile_with(name))
        tight, roomy = compile_with(name, needed), compile_with(name, 2 * needed)
        self.assertEqual(pool(tight), needed)  # exactly enough: every qubit is shared as it must be
        self.assertLessEqual(pool(roomy), 2 * needed)
        self.assertGreater(pool(roomy), needed)  # the spare qubits are used
        self.assertLess(metrics.of_qc(roomy).depth, metrics.of_qc(tight).depth)  # and so less waits for a qubit that is being cleared
        self.assertLess(outcome(name, roomy), TOLERANCE)

    def test_a_die_that_feeds_the_result_through_a_flag_is_held_even_when_the_flag_is_released_and_is_reported_as_garbage(self):
        name = "conditional on a distribution"  # a = 0..4; x = 2; if a < 2 { x += 3; } else { x -= 1; } return x;
        program = translate(parse(entry(name).source, "v2"))
        for strategy in ("lazy", "eager"):
            qc = pipeline.optimise(program, *tuple(p.replace("strategy=lazy", f"strategy={strategy}") for p in pipeline.DEFAULT))
            garbage = metrics.garbage(qc)
            entangled = [register for register in garbage.registers if "entangled" in register[2]]
            self.assertEqual([qubits for _, qubits, _ in entangled], [3], strategy)  # the die of 0..4, however its flag is handled
            self.assertEqual(garbage.blocked_qubits, 3, strategy)
            self.assertLess(outcome(name, qc), TOLERANCE, strategy)

    def test_with_no_system_everything_that_could_be_released_is_reported_as_held_only_because_it_was_not_needed(self):
        qc = compile_with("a loop variable compared with a distribution")
        garbage = metrics.garbage(qc)
        self.assertGreater(garbage.optional_qubits, 0)
        self.assertEqual(sum(q for _, q, reason in garbage.registers if reason.startswith("could be released")), garbage.optional_qubits)
        self.assertIn("released_registers = 0", qc)

    def test_independent_regions_run_side_by_side_when_the_system_has_the_qubits_and_use_fewer_than_a_qubit_for_every_register(self):
        # x and y are accumulated from two independent dice and joined at the end; with the least qubits they follow one another
        name = "two independent accumulators in a loop joined by a sum"
        program = translate(parse(entry(name).source, "v2"))
        unpooled = metrics.of_qc(pipeline.optimise(program, *(p for p in pipeline.DEFAULT if p != "--assign-qubits")))
        least = compile_with(name)
        roomy = compile_with(name, math.ceil(1.5 * pool(least)))
        blindly = compile_with(name, math.ceil(1.5 * pool(least)), policy="spread")
        self.assertEqual(metrics.of_qc(roomy).depth, unpooled.depth)  # as shallow as with a qubit of its own for every register
        self.assertLess(metrics.of_qc(roomy).qubits, unpooled.qubits)  # on far fewer qubits
        self.assertGreater(metrics.of_qc(least).depth, metrics.of_qc(roomy).depth)  # which the least qubits cannot do
        self.assertLessEqual(metrics.of_qc(roomy).qubits, metrics.of_qc(blindly).qubits)  # spending spare qubits only where reuse would wait
        self.assertLess(outcome(name, roomy), TOLERANCE)

    def test_the_order_of_independent_operations_is_chosen_so_that_less_waits_for_a_qubit_on_the_same_qubits(self):
        name = "two independent conditional updates joined by a sum"
        program = pipeline.with_system(translate(parse(entry(name).source, "v2")), System(qubits=26))
        as_written = tuple(p for p in pipeline.DEFAULT if not p.startswith("--reorder-ops"))
        written, chosen = (pipeline.optimise(program, *passes) for passes in (as_written, pipeline.DEFAULT))
        self.assertLessEqual(pool(chosen), pool(written))
        self.assertLess(metrics.of_qc(chosen).depth, metrics.of_qc(written).depth)  # the chain with more work still to come goes first
        self.assertLess(outcome(name, chosen), TOLERANCE)


if __name__ == "__main__":
    unittest.main()

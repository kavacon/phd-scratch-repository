import re
import unittest

from compiler.parser.parser import parse
from experiment_existing_ir import pipeline
from experiment_existing_ir.translate import translate
from tests.golden import GOLDEN


def planned(source: str, *passes: str) -> str:
    return pipeline.optimise(translate(parse(source, "v2")), *pipeline.BEFORE_LOWERING, "--plan-registers", *passes)


@unittest.skipUnless(pipeline.built(), "existing-ir-opt is not built: run helper/cpp/setup.sh and cpp/build.sh")
class Registers(unittest.TestCase):
    def test_updates_done_in_place_share_the_register_of_their_target_and_its_width_is_what_the_ranges_need(self):
        text = planned("a = 0..4; b = a + 3; b += 1; return b;")
        updates = [line for line in text.splitlines() if "add_assign" in line]
        self.assertEqual(len(updates), 2)
        registers = {re.search(r"existing_ir\.register = array<i64: (\d+)>", line).group(1) for line in updates}
        self.assertEqual(len(registers), 1)  # the range and both updates are one register
        self.assertTrue(all("existing_ir.in_place" in line and "existing_ir.qubits = array<i64: 4>" in line for line in updates))  # 0..3 + 4 is 4..7

    def test_a_register_is_definite_unless_a_value_in_it_depends_on_a_distribution_and_a_constant_used_only_as_an_operand_has_none(self):
        text = pipeline.optimise(translate(parse("a = 0..4; b = 3; c = a + b; return c;", "v2")), *pipeline.BEFORE_LOWERING,
                                 "--dist-status", "--plan-registers")
        constant = next(line for line in text.splitlines() if "arith.constant" in line)
        self.assertIn("existing_ir.immediate", constant)  # it is written into the operation that uses it
        self.assertIn("existing_ir.register = array<i64: -1>", constant)
        status = {}
        for line in text.splitlines():
            register = re.search(r"existing_ir\.register = array<i64: (\d+)>", line)
            if register:
                status[int(register.group(1))] = int(re.search(r"existing_ir\.status = array<i64: (\d)>", line).group(1))
        self.assertEqual(list(status.values()), [1])  # the die and its sum are one superposed register

    def test_releasing_each_register_at_its_last_use_needs_fewer_qubits_than_holding_them_all(self):
        text = planned("a = 0..4; b = a + 1; c = b - a; d = c + c; return d;", "--qubit-liveness")
        held = int(re.search(r"existing_ir\.logical_qubits = (\d+)", text).group(1))
        peak = int(re.search(r"existing_ir\.peak_logical_qubits = (\d+)", text).group(1))
        self.assertLess(peak, held)
        # the lowering holds every register to the return and adds the temporaries of each operation, so it needs more than either
        built = pipeline.build(translate(parse("a = 0..4; b = a + 1; c = b - a; d = c + c; return d;", "v2")), *pipeline.DEFAULT)
        self.assertGreater(int(re.search(r"existing_ir\.peak_qubits = (\d+)", built).group(1)), held)

    def test_the_planned_qubits_are_exactly_what_the_lowering_allocates_and_each_way_of_providing_scratch_needs_no_more_than_the_last(self):
        for golden in GOLDEN:
            program = translate(parse(golden.source, "v2"))
            plan = pipeline.optimise(program, *pipeline.BEFORE_LOWERING, "--plan-registers", "--qubit-liveness")
            held, dedicated, freed, pool = (int(re.search(rf"existing_ir\.{key} = (\d+)", plan).group(1))
                                            for key in ("qubits_held", "qubits_dedicated", "qubits_scratch_freed", "qubits_pool"))
            lowered = int(re.search(r"existing_ir\.peak_qubits = (\d+)", pipeline.build(program, *pipeline.DEFAULT)).group(1))
            self.assertEqual(freed, lowered, golden.name)  # registers held to the return, each operation's temporaries freed
            self.assertLessEqual(pool, freed, golden.name)
            self.assertLessEqual(freed, dedicated, golden.name)
            self.assertLessEqual(dedicated, held, golden.name)

    def test_a_value_computed_early_is_moved_to_just_before_its_use_and_the_program_needs_fewer_qubits(self):
        # b and e are made first and used last; the square in between is where the most qubits are needed, and does not use them
        text = """
func.func @main() -> i32 {
  %b = dist.uniform 0, 16 : !dist.dist<i32>
  %e = dist.uniform 0, 2 : !dist.dist<i32>
  %d = dist.uniform 0, 16 : !dist.dist<i32>
  %sq = dist.muli %d, %d : !dist.dist<i32>, !dist.dist<i32> -> !dist.dist<i32>
  %m1 = dist.measure %sq : !dist.dist<i32> -> i32
  %t = dist.add_assign %b, %e : !dist.dist<i32>, !dist.dist<i32> -> !dist.dist<i32>
  %m = dist.measure %t : !dist.dist<i32> -> i32
  return %m : i32
}"""
        lines = lambda extra: pipeline.optimise(text, *extra).splitlines()
        position = lambda ls, name: next(i for i, line in enumerate(ls) if name in line)
        plain, sunk = lines(()), lines(("--sink-to-use",))
        self.assertLess(position(plain, "dist.uniform 0, 16"), position(plain, "dist.muli"))  # b is made first
        self.assertGreater(max(i for i, line in enumerate(sunk) if "dist.uniform 0, 16" in line), position(sunk, "dist.muli"))  # and now after
        used = lambda *extra: int(re.search(r"pool_qubits = (\d+)", pipeline.optimise(
            text, *extra, "--plan-registers", "--lower-to-qc", "--release-registers", "--assign-qubits")).group(1))
        self.assertLess(used("--sink-to-use"), used())

    def test_a_value_does_not_move_past_the_update_of_one_of_its_inputs(self):
        # t reads a before the update changes it, so it stays before the update even though its result is used after it
        text = """
func.func @main() -> i32 {
  %five = arith.constant 5 : i32
  %a = dist.uniform 0, 4 : !dist.dist<i32>
  %t = dist.addi %a, %five : !dist.dist<i32>, i32 -> !dist.dist<i32>
  %a2 = dist.add_assign %a, %five : !dist.dist<i32>, i32 -> !dist.dist<i32>
  %u = dist.addi %t, %a2 : !dist.dist<i32>, !dist.dist<i32> -> !dist.dist<i32>
  %m = dist.measure %u : !dist.dist<i32> -> i32
  return %m : i32
}"""
        sunk = pipeline.optimise(text, "--sink-to-use")
        self.assertLess(sunk.index("dist.addi %"), sunk.index("dist.add_assign"))


if __name__ == "__main__":
    unittest.main()

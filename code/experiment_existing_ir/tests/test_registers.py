import re
import subprocess
import unittest

from compiler.parser.parser import parse
from compiler.system import System
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
            held, dedicated, pool = (int(re.search(rf"existing_ir\.{key} = (\d+)", plan).group(1))
                                     for key in ("qubits_held", "qubits_dedicated", "qubits_pool"))
            lowered = int(re.search(r"existing_ir\.peak_qubits = (\d+)", pipeline.build(program, *pipeline.DEFAULT)).group(1))
            self.assertEqual(held, lowered, golden.name)
            self.assertLessEqual(pool, dedicated, golden.name)
            self.assertLessEqual(dedicated, held, golden.name)

    def test_a_register_is_released_only_when_the_budget_needs_its_qubits_and_a_program_that_cannot_fit_is_refused(self):
        program = translate(parse("a = 0..4; b = a + 1; c = b - a; d = c + c; return d;", "v2"))

        def classified(qubits=None):
            text = program if qubits is None else pipeline.with_system(program, System(qubits=qubits))
            out = pipeline.optimise(text, *pipeline.BEFORE_LOWERING, "--plan-registers", "--classify-release")
            classes = [int(n) for n in re.search(r"release_class = array<i64: ([^>]*)>", out).group(1).split(",")]
            return classes, int(re.search(r"qubits_needed = (\d+)", out).group(1))

        classes, needed = classified()
        self.assertEqual(set(classes), {0})  # no system: nothing must be released
        self.assertEqual(classified(needed)[0], classes)  # exactly enough without releasing
        released, now_needed = classified(needed - 4)
        self.assertIn(1, released)  # tighter: some dead register is released at its last use
        self.assertLessEqual(now_needed, needed - 4)
        with self.assertRaises(subprocess.CalledProcessError) as raised:
            classified(5)
        self.assertIn("even with every dead register released", raised.exception.stderr)
        # the plan says this budget is feasible by releasing registers; the lowering does not release yet, so it still refuses
        with self.assertRaises(subprocess.CalledProcessError) as lowering:
            pipeline.build(program, *pipeline.DEFAULT, system=System(qubits=needed - 4))
        self.assertIn("qubits at once but the system has", lowering.exception.stderr)


if __name__ == "__main__":
    unittest.main()

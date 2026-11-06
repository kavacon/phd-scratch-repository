import re
import unittest

from compiler.parser.parser import parse
from experiment_existing_ir import pipeline
from experiment_existing_ir.translate import translate


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


if __name__ == "__main__":
    unittest.main()

import importlib.util
import unittest

from experiments.allocation_poc.programs import PROGRAMS
from checking import qiskit_tools

HAVE_QRISP = importlib.util.find_spec("qrisp") is not None and qiskit_tools.qiskit_available()
needs_qrisp = unittest.skipUnless(HAVE_QRISP, "Qrisp and Qiskit are not installed")


class QrispProgramsTestCase(unittest.TestCase):
    def test_every_demo_program_has_a_qrisp_version_with_a_main(self):
        from experiments.allocation_poc import qrisp_programs
        self.assertEqual({p["id"] for p in PROGRAMS}, set(qrisp_programs.SOURCES))
        for program_id, source in qrisp_programs.SOURCES.items():
            self.assertIn("def main():", source, program_id)
            self.assertIn("@auto_uncompute", source, program_id)
            self.assertNotIn("cuccaro", source, program_id)  # built-in operators only
            self.assertIn("from qrisp import QuantumFloat, auto_uncompute", source, program_id)


@needs_qrisp
class ComparisonTestCase(unittest.TestCase):
    def test_a_comparison_has_both_sources_and_counts_text_and_a_drawing_for_all_four_circuits(self):
        from experiments.allocation_poc import qrisp_comparison as comparison
        built = comparison.build("expression", 4)
        self.assertIn("function f", built.sources["ours"])
        self.assertIn("@auto_uncompute", built.sources["qrisp"])
        self.assertIn("WIDTH = 4", built.sources["qrisp"])
        for side in comparison.SIDES:
            for variant in comparison.VARIANTS:
                entry = built.sides[side]
                self.assertGreater(comparison.metrics(built, side, variant)["gates"], 0, (side, variant))
                self.assertIn("OPENQASM", entry.text[variant])
                self.assertTrue(entry.drawings[variant]["gates"], (side, variant))
        self.assertLessEqual(set(comparison.metrics(built, "ours", "optimised")["ops"]), {"u", "cx", "measure"})


if __name__ == '__main__':
    unittest.main()

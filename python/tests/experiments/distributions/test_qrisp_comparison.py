import importlib.util
import unittest

from checking import qiskit_tools
from experiments.distributions.programs import PROGRAMS

HAVE_QRISP = importlib.util.find_spec("qrisp") is not None and qiskit_tools.qiskit_available() and qiskit_tools.aer_available()
needs_qrisp = unittest.skipUnless(HAVE_QRISP, "Qrisp, Qiskit and Qiskit Aer are not installed")


class QrispProgramsTestCase(unittest.TestCase):
    def test_every_program_has_a_qrisp_version_with_a_main_that_uses_only_qrisp_operators(self):
        from experiments.distributions import qrisp_programs
        self.assertEqual({p["id"] for p in PROGRAMS}, set(qrisp_programs.SOURCES))
        for program_id, source in qrisp_programs.SOURCES.items():
            self.assertIn("def main():", source, program_id)
            self.assertIn("@auto_uncompute", source, program_id)
            self.assertNotIn("cuccaro", source, program_id)


@needs_qrisp
class ComparisonTestCase(unittest.TestCase):
    def test_both_circuits_return_the_distribution_the_instruction_machine_gives_and_all_four_have_counts(self):
        from experiments.distributions import qrisp_comparison as comparison
        built = comparison.build("branch-nested", 8)
        self.assertEqual({0: 0.5, 1: 0.25, 3: 0.25}, built.expected)
        for side in comparison.SIDES:
            for variant in comparison.VARIANTS:
                self.assertGreater(comparison.metrics(built, side, variant)["gates"], 0, (side, variant))
            report = comparison.run_circuit("branch-nested", 8, side, "optimised", 500)
            self.assertTrue(report["matches"], (side, report["distance"]))

    def test_the_demo_describes_the_comparison_in_known_blocks_that_serialise(self):
        import json
        from demo.model import to_data
        from experiments.distributions.demos import compare
        result = compare.run(compare.DEMO.clean_inputs({"case": "branch-copy"}), [])
        self.assertEqual(["programs", "statistics", "circuits", "state"], [stage.id for stage in result.stages])
        json.dumps(to_data(result))

    def test_the_final_states_show_what_each_side_releases_and_ours_agrees_with_the_instruction_machine(self):
        import json
        from demo.model import to_data
        from experiments.distributions import qrisp_comparison as comparison
        from experiments.distributions.demos import compare
        ours, theirs = (comparison.final_state("shared", 8, side) for side in comparison.SIDES)
        self.assertTrue(ours["reference"]["matches"])
        self.assertEqual({}, ours["held"])  # the input distributions are released by undoing them
        self.assertFalse(ours["result_entangled"])
        self.assertTrue(any(not info["entangled"] for info in theirs["held"].values()))  # a register Qrisp holds that could be undone
        inputs = compare.DEMO.clean_inputs({"case": "shared"})
        json.dumps([to_data(block) for block in compare.state(inputs, [], {"side": "qrisp"})])


if __name__ == '__main__':
    unittest.main()

import json
import unittest

from checking.simulate import simulate_isa
from demo.model import to_data, walk
from experiments.distributions import experiment
from experiments.distributions.demos import explorer
from experiments.distributions.programs import PROGRAMS
from experiments.distributions.views import entangled_groups, groups_by_joint_operation

KNOWN = {"code", "diff", "metrics", "stats", "switcher", "grid", "timeline", "note", "action", "diagram", "columns"}


class ExplorerTestCase(unittest.TestCase):
    def test_every_program_describes_to_known_blocks_that_serialise_and_ends_with_the_optimised_circuit(self):
        for case in explorer.DEMO.cases:
            with self.subTest(program=case.id):
                result = explorer.run(explorer.DEMO.clean_inputs({"case": case.id}), [s.id for s in explorer.DEMO.steps])
                self.assertIsNone(result.error)
                self.assertEqual(["source", "optimise"], [result.stages[0].id, result.stages[-1].id])
                for block in walk([block for stage in result.stages for block in stage.blocks]):
                    self.assertIn(to_data(block)["type"], KNOWN)
                json.dumps(to_data(result))

    def test_the_steps_are_the_passes(self):
        self.assertEqual([fn.compile_pass.name for fn in experiment.compiler().passes], [s.id for s in explorer.DEMO.steps])


class EntanglementGroupsTestCase(unittest.TestCase):
    def test_the_state_separates_registers_the_joint_operation_rule_joins(self):
        # c is the XOR of two independent bits: no two registers are correlated on their own, but all three are entangled
        state = {(a, b, a ^ b, 7): 0.5 for a in (0, 1) for b in (0, 1)}
        self.assertEqual([["a", "b", "c"], ["d"]], entangled_groups(("a", "b", "c", "d"), state))
        source = next(p["source"] for p in PROGRAMS if p["id"] == "mixed")
        unoptimised = experiment.compiler().run(
            source, until="lower_to_isa", skip=["collapse_updates", "fold_constants", "drop_unread"]).output
        self.assertEqual(1, len(groups_by_joint_operation(unoptimised)))
        self.assertEqual(
            [["a#1", "b#1", "d#1"], ["c#1"], ["k#1"], ["m#1"]],
            entangled_groups(*simulate_isa(unoptimised, experiment.SYSTEM).state))


if __name__ == '__main__':
    unittest.main()

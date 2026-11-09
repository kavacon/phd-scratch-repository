import unittest

from compiler.parser.parser import parse
from experiment_existing_ir import comparison, metrics, optimisers, pipeline
from experiment_existing_ir.translate import translate
from tests.golden import TOLERANCE, entry

PRODUCT = "a product of a distribution and an integer"


@unittest.skipUnless(pipeline.built(), "existing-ir-opt is not built: run helper/cpp/setup.sh and cpp/build.sh")
class Optimisers(unittest.TestCase):
    def test_every_optimiser_keeps_the_outcome_and_the_t_cost_counts_phases_however_they_are_written(self):
        qc = pipeline.optimise(translate(parse(entry(PRODUCT).source, "v2")), *optimisers.unpooled(pipeline.DEFAULT))
        reference = {}
        costs = {}
        for name, run in optimisers.OPTIMISERS.items():
            circuit = run(qc)
            counts, _ = comparison.outcomes(circuit)
            reference = reference or counts
            self.assertLess(comparison.distance(counts, {k: v / sum(reference.values()) for k, v in reference.items()}), TOLERANCE, name)
            costs[name] = metrics.of_clifford_t(circuit)
        # a Toffoli is seven T gates, and ZX writes its T gates as phases of pi/4: counting only gates named T would read none
        self.assertEqual(costs["none"].t_count % 7, 0)
        self.assertTrue(0 < costs["pyzx"].t_count < costs["none"].t_count)
        self.assertLessEqual(costs["pyzx"].t_depth, costs["pyzx"].t_count)

    def test_the_comparison_measures_a_program_with_and_without_the_source_rewrites(self):
        source = "a = 0..4; b = 0..3; x = 0; if a < 2 { x = b * b + 1; } else { x = b * b + 2; } return x;"
        cells = {(c.rewrites, c.optimiser): c for c in comparison.measure("arms", source, "rewrites") + comparison.measure("arms", source, "no rewrites")
                 if c.optimiser == "none"}
        self.assertLess(cells[("rewrites", "none")].cost[1], cells[("no rewrites", "none")].cost[1])  # fewer gates with the rewrites


if __name__ == "__main__":
    unittest.main()

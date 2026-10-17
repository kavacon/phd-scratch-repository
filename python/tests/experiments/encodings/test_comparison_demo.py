import json
import unittest
from unittest import mock

from demo.model import to_data
from experiments.encodings import experiment
from experiments.encodings.demos import comparison
from experiments.encodings.encodings import DEFAULT


class ComparisonDemoTestCase(unittest.TestCase):
    def run_case(self, case, enabled=DEFAULT, **inputs):
        return comparison.run(comparison.DEMO.clean_inputs({"case": case, **inputs}), list(enabled))

    def test_a_program_shows_its_source_a_results_table_per_encoding_and_the_circuits(self):
        result = self.run_case("mixed")
        self.assertIsNone(result.error)
        self.assertEqual(["source", "results", "circuits"], [s.id for s in result.stages])
        results = next(s for s in result.stages if s.id == "results")
        tables = [b for b in results.blocks if to_data(b)["type"] == "stats"]
        self.assertEqual(len(DEFAULT), len(tables))
        self.assertTrue(all(row.cells[1].text == "yes" for table in tables for row in table.rows))  # every circuit agrees
        json.dumps(to_data(result))

    def test_the_all_programs_view_has_a_table_per_measure_with_the_optimisers_side_by_side(self):
        with mock.patch.object(experiment, "zx_available", return_value=False):
            result = self.run_case("all", enabled=["binary", "fourier"], register_width=3)
        self.assertEqual(["suite"], [s.id for s in result.stages])
        tables = [b for b in result.stages[0].blocks if to_data(b)["type"] == "stats"]
        self.assertEqual(len(comparison.MEASURES), len(tables))
        self.assertEqual(13, len(tables[0].rows))


if __name__ == '__main__':
    unittest.main()

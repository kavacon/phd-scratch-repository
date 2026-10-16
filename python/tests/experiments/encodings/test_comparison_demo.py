import json
import unittest

from demo.model import to_data, walk
from experiments.encodings.demos import comparison
from experiments.encodings.encodings import DEFAULT, ENCODINGS
from experiments.encodings.programs import PROGRAMS

KNOWN = {"code", "diff", "metrics", "stats", "switcher", "grid", "timeline", "note", "action", "diagram", "columns"}


def blocks_of(result):
    return list(walk([block for stage in result.stages for block in stage.blocks]))


class ComparisonDemoTestCase(unittest.TestCase):
    def run_case(self, case, enabled=DEFAULT, **inputs):
        return comparison.run(comparison.DEMO.clean_inputs({"case": case, **inputs}), list(enabled))

    def test_every_program_describes_to_known_blocks_that_serialise(self):
        for program in PROGRAMS:
            with self.subTest(program=program["id"]):
                result = self.run_case(program["id"])
                self.assertIsNone(result.error)
                self.assertEqual(["source", "results", "circuits"], [s.id for s in result.stages])
                for block in blocks_of(result):
                    self.assertIn(to_data(block)["type"], KNOWN)
                json.dumps(to_data(result))

    def test_the_steps_are_the_encodings_with_the_default_pair_switched_on(self):
        steps = comparison.DEMO.steps
        self.assertEqual(list(ENCODINGS), [s.id for s in steps])
        self.assertEqual(list(DEFAULT), [s.id for s in steps if s.enabled])

    def test_the_results_have_a_table_per_encoding_with_a_row_per_circuit_stage_and_all_agree(self):
        result = self.run_case("mixed")
        results = next(s for s in result.stages if s.id == "results")
        tables = [b for b in results.blocks if to_data(b)["type"] == "stats"]
        self.assertEqual(len(DEFAULT), len(tables))
        for table in tables:
            self.assertEqual(2, len(table.rows))
            self.assertTrue(all(row.cells[1].text == "yes" for row in table.rows))

    def test_the_suite_compares_every_program_on_each_measure_with_a_column_per_encoding(self):
        result = self.run_case("all")
        self.assertEqual(["suite"], [s.id for s in result.stages])
        suite = result.stages[0]
        tables = [b for b in suite.blocks if to_data(b)["type"] == "stats"]
        self.assertEqual(len(comparison.MEASURES), len(tables))
        for table in tables:
            self.assertEqual(len(PROGRAMS), len(table.rows))
            self.assertEqual(1 + len(DEFAULT), len(table.headers))

    def test_an_encoding_that_is_not_built_yet_is_reported_beside_the_others(self):
        result = self.run_case("wrap", enabled=["binary", "angle"])
        results = next(s for s in result.stages if s.id == "results")
        notes = [to_data(b) for b in results.blocks if to_data(b)["type"] == "note" and to_data(b)["tone"] == "bad"]
        self.assertEqual(1, len(notes))
        self.assertIn("NotImplementedError", notes[0]["text"])
        self.assertTrue(any(to_data(b)["type"] == "stats" for b in results.blocks))

    def test_nothing_enabled_says_so_and_a_syntax_error_is_reported_against_the_input(self):
        self.assertEqual("none", self.run_case("wrap", enabled=[]).stages[0].id)
        error = comparison.run(comparison.DEMO.clean_inputs({"source": "x = ;"}), ["binary"]).error
        self.assertTrue(error.message)

    def test_the_register_width_is_kept_in_range(self):
        self.assertEqual((8, 2), tuple(comparison.DEMO.clean_inputs({"register_width": w})["register_width"] for w in (99, 0)))


if __name__ == '__main__':
    unittest.main()

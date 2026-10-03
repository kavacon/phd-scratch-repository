import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from allocation_poc.demo import PASSES, Handler, diff_lines, run_pipeline
from allocation_poc.programs import PROGRAMS

ALL = [p.id for p in PASSES]


class DiffTestCase(unittest.TestCase):
    def kinds(self, before, after):
        old, new = diff_lines(before, after)
        return [l["kind"] for l in old], [l["kind"] for l in new]

    def test_unchanged_added_and_removed_lines(self):
        self.assertEqual((["same", "same"], ["same", "added", "same"]), self.kinds(["a", "c"], ["a", "b", "c"]))
        self.assertEqual((["same", "removed", "same"], ["same", "same"]), self.kinds(["a", "b", "c"], ["a", "c"]))

    def test_a_modified_line_highlights_only_what_changed(self):
        old, new = diff_lines(["    x = x + 1;"], ["    x#2 = x#1 + 1;"])
        self.assertEqual("changed", new[0]["kind"])
        text = new[0]["text"]
        self.assertEqual(["x#2", "x#1"], [text[s:e] for s, e in new[0]["spans"]])
        self.assertEqual(["x", "x"], [old[0]["text"][s:e] for s, e in old[0]["spans"]])

    def test_lines_that_are_too_different_are_not_called_modified(self):
        self.assertEqual((["removed"], ["added"]), self.kinds(["return a;"], ["    p#1 ~= w#0 * h#0;"]))


class PipelineTestCase(unittest.TestCase):
    def test_stages_follow_compiler_order_and_skip_passes_that_are_off(self):
        stages = run_pipeline(PROGRAMS[0]["source"], ["allocation", "balance"])["stages"]
        self.assertEqual(["source", "balance", "allocation"], [s["id"] for s in stages])

    def test_the_source_stage_has_nothing_to_compare_against(self):
        source = run_pipeline(PROGRAMS[0]["source"], [])["stages"][0]
        self.assertEqual([], source["before"])
        self.assertTrue(all(line["kind"] == "same" for line in source["after"]))

    def test_each_pass_reports_what_it_changed(self):
        stages = {s["id"]: s for s in run_pipeline(PROGRAMS[0]["source"], ALL)["stages"]}
        self.assertEqual(7, stages["rename"]["stats"]["changed"])
        self.assertEqual(3, stages["allocation"]["stats"]["releases"])  # p in area, then y and x in the caller

    def test_every_preloaded_program_keeps_its_result_and_ends_without_leftovers(self):
        for program in PROGRAMS:
            with self.subTest(program=program["id"]):
                stages = run_pipeline(program["source"], ALL)["stages"]
                self.assertEqual(["source", "balance", "rename", "allocation"], [s["id"] for s in stages])
                self.assertFalse(any(s.get("error") for s in stages))
                self.assertEqual(1, len({s["run"]["value"] for s in stages}))
                if program["id"] != "loop":  # a value carried round a loop is not released
                    self.assertEqual([], stages[-1]["run"]["leaked"])

    def test_early_release_lowers_peak_width(self):
        width = next(p for p in PROGRAMS if p["id"] == "width")
        stages = run_pipeline(width["source"], ALL)["stages"]
        self.assertLess(stages[-1]["run"]["peak"], stages[-2]["run"]["peak"])

    def test_a_pass_that_fails_stops_the_pipeline_and_says_why(self):
        source = "function f(a: int): int { if a < 2 { a = 1; } return a; } return f(1);"
        stages = run_pipeline(source, ["rename", "allocation"])["stages"]
        self.assertEqual(["source", "rename"], [s["id"] for s in stages])
        self.assertIn("balance", stages[-1]["error"])

    def test_syntax_errors_are_reported_with_a_position(self):
        error = run_pipeline("function f(): int { return 1 + ; } return f();", ALL)["error"]
        self.assertEqual((1, "Unexpected ';'."), (error["line"], error["message"]))
        self.assertIn("return", run_pipeline("x = 1;", ALL)["error"]["message"])

    def test_a_variable_loop_bound_explains_the_rule(self):
        error = run_pipeline("function f(n: int): int { for i in 0 .. n { } return n; } return f(1);", ALL)["error"]
        self.assertIn("whole-number literals", error["message"])

    def test_programs_that_cannot_run_are_shown_anyway(self):
        stages = run_pipeline("function f(n: int): int { return g(n); } return f(1);", ALL)["stages"]
        self.assertIn("not defined", stages[0]["run"]["error"])
        endless = "function f(n: int): int { x = n; for i in 0 .. 10000000 { x = x + 1; } return x; } return f(1);"
        self.assertIn("step limit", run_pipeline(endless, [])["stages"][0]["run"]["error"])

    def test_a_loop_carried_value_that_is_not_returned_is_reported(self):
        source = "function g(n: int): int { s = 0; for i in 0 .. 3 { s = s + i; } return n; } return g(3);"
        notes = run_pipeline(source, ALL)["stages"][-1]["notes"]
        self.assertTrue(any("s#2" in note for note in notes))


class LivenessViewTestCase(unittest.TestCase):
    def view(self, program_id):
        program = next(p for p in PROGRAMS if p["id"] == program_id)
        stages = run_pipeline(program["source"], ALL)["stages"]
        return stages[-1]

    def test_only_the_release_stage_carries_a_liveness_view(self):
        stages = run_pipeline(PROGRAMS[0]["source"], ALL)["stages"]
        self.assertEqual([None, None, None], [s["liveness"] for s in stages[:3]])
        self.assertIsNotNone(stages[3]["liveness"])

    def test_scopes_follow_the_structure_of_the_program(self):
        scopes = self.view("if-else")["liveness"]["scopes"]
        self.assertEqual(["Program", "function f", "then branch of row 3", "else branch of row 3"], [s["title"] for s in scopes])
        self.assertEqual([0, 0, 1, 1], [s["depth"] for s in scopes])

    def test_columns_say_where_each_version_is_live_and_released(self):
        function = {c["name"]: c for c in self.view("if-else")["liveness"]["scopes"][1]["columns"]}
        self.assertEqual("param", function["x#0"]["kind"])
        self.assertEqual("output", function["r#1"]["kind"])
        a = function["a#1"]
        self.assertEqual((0, 3, 4, 3), (a["defined"], a["live_end"], a["conservative_end"], a["release_row"]))
        # y#2 and a#1 are released at the same point, y#2 first
        self.assertEqual((1, 2), (function["y#2"]["order"], a["order"]))

    def test_a_branch_marks_what_leaves_it_as_returned(self):
        then_branch = {c["name"]: c for c in self.view("if-else")["liveness"]["scopes"][2]["columns"]}
        self.assertEqual("temp", then_branch["t#1"]["kind"])
        self.assertEqual("output", then_branch["y#2"]["kind"])

    def test_every_released_version_shown_in_the_view_is_released_in_the_output(self):
        # the view and the inserted code come from separate steps, so this checks they agree
        for program in PROGRAMS:
            with self.subTest(program=program["id"]):
                stage = run_pipeline(program["source"], ALL)["stages"][-1]
                released = [line for line in stage["text"].splitlines() if "~=" in line]
                for scope in stage["liveness"]["scopes"]:
                    for column in scope["columns"]:
                        if column["kind"] == "temp":
                            self.assertTrue(any(f"{column['name']} ~=" in line for line in released), column["name"])

    def test_loop_carried_values_are_shown_as_kept(self):
        scopes = self.view("loop")["liveness"]["scopes"]
        kinds = {c["name"]: c["kind"] for c in scopes[1]["columns"]}
        self.assertEqual("kept", kinds["s#1"])

    def test_the_view_is_skipped_unless_rename_ran(self):
        stages = run_pipeline(PROGRAMS[0]["source"], ["balance", "allocation"])["stages"]
        self.assertIsNone(stages[-1]["liveness"])


class ServerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def get(self, path):
        with urllib.request.urlopen(self.url + path) as response:
            return response.read()

    def test_serves_the_page_programs_and_passes(self):
        self.assertIn(b"Pipeline Explorer", self.get("/"))
        self.assertEqual(len(PROGRAMS), len(json.loads(self.get("/api/programs"))))
        self.assertEqual(ALL, [p["id"] for p in json.loads(self.get("/api/passes"))])

    def test_runs_a_program(self):
        request = urllib.request.Request(
            self.url + "/api/run",
            json.dumps({"source": PROGRAMS[0]["source"], "passes": ALL}).encode(),
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            stages = json.loads(response.read())["stages"]
        self.assertEqual(4, len(stages))


if __name__ == '__main__':
    unittest.main()

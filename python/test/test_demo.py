import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from allocation_poc.demo import PASSES, Handler, clean_options, diff_lines, run_pipeline, source_origin
from allocation_poc.programs import PROGRAMS

ALL = [p.id for p in PASSES]
RELEASES = [i for i in ALL if i not in ("inline", "isa", "allocate")]  # up to releases, with functions kept as scopes


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


class InlineStageTestCase(unittest.TestCase):
    def test_the_inline_stage_removes_functions_and_keeps_the_result(self):
        for program in PROGRAMS:
            stages = {s["id"]: s for s in run_pipeline(program["source"], ALL)["stages"]}
            self.assertNotIn("function", stages["inline"]["text"], program["id"])
            self.assertEqual(stages["source"]["run"]["value"], stages["inline"]["run"]["value"], program["id"])
            self.assertEqual(["source", "inline"], [s for s in stages][:2])

    def test_turning_inline_off_makes_the_instruction_stage_say_functions_must_be_inlined(self):
        stages = run_pipeline(PROGRAMS[0]["source"], [i for i in ALL if i != "inline"])["stages"]
        self.assertIn("inlined", stages[-1]["error"])

class AllocationStageTestCase(unittest.TestCase):
    def stage(self, program_id, width=1, qubits=16):
        source = next(p for p in PROGRAMS if p["id"] == program_id)["source"]
        return run_pipeline(source, ALL, {"register_width": width, "qubits": qubits})["stages"][-1]

    def test_the_last_stage_is_the_allocation_with_the_machine_it_was_given(self):
        stage = self.stage("if-else", width=2, qubits=14)
        self.assertEqual("allocate", stage["id"])
        view = stage["allocation"]
        self.assertEqual((2, 14, 6, 5, True), (view["register_width"], view["qubits"], view["slots"], view["needed"], view["fits"]))
        self.assertEqual((2, 12), (view["scratch"], view["qubits_used"]))  # 5 registers of 2, and 2 for scratch
        self.assertIn("R0", stage["text"])
        self.assertEqual([], stage["run"]["leaked"])
        self.assertNotIn("error", stage)

    def test_the_result_is_unchanged_by_allocation_for_every_program(self):
        for program in PROGRAMS:
            stages = {s["id"]: s for s in run_pipeline(program["source"], ALL)["stages"]}
            self.assertEqual(stages["isa"]["run"]["value"], stages["allocate"]["run"]["value"], program["id"])

    def test_too_few_qubits_is_reported_but_the_packing_is_still_shown(self):
        stage = self.stage("if-else", width=4, qubits=16)
        self.assertIn("needs 5 register slots but only 3 are available", stage["error"])
        self.assertFalse(stage["allocation"]["fits"])
        self.assertTrue(stage["allocation"]["steps"])

    def test_every_step_says_what_each_slot_held_and_where_it_came_from_in_the_source(self):
        steps = self.stage("area")["allocation"]["steps"]
        inlined = [h for step in steps for h in step["held"] if h["register"].startswith("area1.p#")]
        self.assertTrue(inlined)
        self.assertEqual(("p", "area", [2]), (inlined[0]["variable"], inlined[0]["function"], inlined[0]["lines"]))  # `p = w * h;`
        caller = [h for step in steps for h in step["held"] if h["register"] == "x#1"]
        self.assertEqual(("x", "main"), (caller[0]["variable"], caller[0]["function"]))
        temporary = [h for step in steps for h in step["held"] if h["register"].startswith("$t")]
        self.assertTrue(not temporary or all(h["variable"] is None and h["lines"] == [] for h in temporary))

    def test_a_registers_origin_is_read_from_its_name(self):
        self.assertEqual(("x", "main"), source_origin("x#2"))
        self.assertEqual(("p", "area"), source_origin("area1.p#1"))
        self.assertEqual(("m", "g"), source_origin("h1.g2.m#1"))
        self.assertEqual(("t", "main"), source_origin("t@3#1"))
        self.assertEqual((None, "main"), source_origin("$t4#1"))

    def test_settings_outside_the_allowed_range_are_pulled_back(self):
        self.assertEqual({"register_width": 1, "qubits": 16}, clean_options(None))
        self.assertEqual({"register_width": 64, "qubits": 0}, clean_options({"register_width": 999, "qubits": -5}))
        self.assertEqual({"register_width": 1, "qubits": 16}, clean_options({"register_width": "x", "qubits": None}))


class PipelineTestCase(unittest.TestCase):
    def test_the_instruction_stage_is_printed_as_instructions_and_run_on_the_instruction_machine(self):
        for program in PROGRAMS:
            stages = run_pipeline(program["source"], ALL)["stages"]
            isa, allocation = stages[-2], stages[-3]
            self.assertEqual(("isa", "allocation"), (isa["id"], allocation["id"]), program["id"])
            self.assertIn("ALLOC", isa["text"])
            self.assertEqual(allocation["run"]["value"], isa["run"]["value"], program["id"])
            self.assertEqual([], isa["run"]["leaked"], program["id"])

    def test_stages_follow_compiler_order_and_skip_passes_that_are_off(self):
        stages = run_pipeline(PROGRAMS[0]["source"], ["allocation", "balance"])["stages"]
        self.assertEqual(["source", "balance", "allocation"], [s["id"] for s in stages])

    def test_the_source_stage_has_nothing_to_compare_against(self):
        source = run_pipeline(PROGRAMS[0]["source"], [])["stages"][0]
        self.assertEqual([], source["before"])
        self.assertTrue(all(line["kind"] == "same" for line in source["after"]))

    def test_each_pass_reports_what_it_changed(self):
        stages = {s["id"]: s for s in run_pipeline(PROGRAMS[0]["source"], RELEASES)["stages"]}
        self.assertEqual((1, 1), (stages["flatten"]["stats"]["changed"], stages["flatten"]["stats"]["added"]))  # the call is returned through a temporary
        self.assertEqual(8, stages["rename"]["stats"]["changed"])
        self.assertEqual(3, stages["allocation"]["stats"]["releases"])  # p in area, then y and x in the caller

    def test_every_preloaded_program_keeps_its_result_and_ends_without_leftovers(self):
        for program in PROGRAMS:
            with self.subTest(program=program["id"]):
                stages = run_pipeline(program["source"], RELEASES)["stages"]
                self.assertEqual(["source", "unroll", "flatten", "balance", "rename", "allocation"], [s["id"] for s in stages])
                self.assertFalse(any(s.get("error") for s in stages))
                self.assertEqual(1, len({s["run"]["value"] for s in stages}))
                self.assertEqual([], stages[-1]["run"]["leaked"])

    def test_early_release_lowers_peak_width(self):
        width = next(p for p in PROGRAMS if p["id"] == "width")
        stages = run_pipeline(width["source"], RELEASES)["stages"]
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



class LivenessViewTestCase(unittest.TestCase):
    def view(self, program_id):
        program = next(p for p in PROGRAMS if p["id"] == program_id)
        stages = run_pipeline(program["source"], RELEASES)["stages"]
        return stages[-1]

    def test_only_the_release_stage_carries_a_liveness_view(self):
        stages = run_pipeline(PROGRAMS[0]["source"], RELEASES)["stages"]
        self.assertEqual([None] * 5, [s["liveness"] for s in stages[:5]])
        self.assertIsNotNone(stages[5]["liveness"])

    def test_scopes_follow_the_structure_of_the_program(self):
        scopes = self.view("if-else")["liveness"]["scopes"]
        self.assertEqual(["Program", "function f", "then branch of row 4", "else branch of row 4"], [s["title"] for s in scopes])
        self.assertEqual([0, 0, 1, 1], [s["depth"] for s in scopes])

    def test_columns_say_where_each_version_is_live_and_released(self):
        function = {c["name"]: c for c in self.view("if-else")["liveness"]["scopes"][1]["columns"]}
        self.assertEqual("param", function["x#0"]["kind"])
        self.assertEqual("output", function["r#1"]["kind"])
        a = function["a#1"]
        self.assertEqual((0, 4, 5, 4), (a["defined"], a["live_end"], a["conservative_end"], a["release_row"]))
        # y#2, the condition register $t1#1 and a#1 are released at the same point, in that order
        self.assertEqual((1, 2, 3), (function["y#2"]["order"], function["$t1#1"]["order"], a["order"]))

    def test_a_branch_marks_what_leaves_it_as_returned(self):
        then_branch = {c["name"]: c for c in self.view("if-else")["liveness"]["scopes"][2]["columns"]}
        self.assertEqual("kept", then_branch["t#1"]["kind"])  # the release of y#2 needs it, so it outlives the branch
        self.assertEqual("output", then_branch["y#2"]["kind"])

    def test_every_released_version_shown_in_the_view_is_released_in_the_output(self):
        # the view and the inserted code come from separate steps, so this checks they agree
        for program in PROGRAMS:
            with self.subTest(program=program["id"]):
                stage = run_pipeline(program["source"], RELEASES)["stages"][-1]
                released = [line for line in stage["text"].splitlines() if "~=" in line]
                for scope in stage["liveness"]["scopes"]:
                    for column in scope["columns"]:
                        if column["kind"] == "temp":
                            self.assertTrue(any(f"{column['name']} ~=" in line for line in released), column["name"])

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
            json.dumps({"source": PROGRAMS[0]["source"], "passes": RELEASES}).encode(),
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            stages = json.loads(response.read())["stages"]
        self.assertEqual(6, len(stages))


if __name__ == '__main__':
    unittest.main()

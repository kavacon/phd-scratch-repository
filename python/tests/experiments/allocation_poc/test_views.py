import unittest

from compiler.system import System
from experiments.allocation_poc.experiment import FRONT_END, compile_source, compiler, text_of
from experiments.allocation_poc.programs import PROGRAMS
from experiments.allocation_poc.views import allocation_view, liveness_view, source_origin

KEPT = ("unroll", "flatten", "balance", "rename")  # up to what the release pass is given, with functions kept as scopes


def program(program_id):
    return next(p for p in PROGRAMS if p["id"] == program_id)["source"]


class LivenessViewTestCase(unittest.TestCase):
    def view(self, program_id):
        return liveness_view(compile_source(program(program_id), FRONT_END[1:-1]))

    def test_scopes_follow_the_structure_of_the_program(self):
        scopes = self.view("if-else")["scopes"]
        self.assertEqual(["Program", "function f", "then branch of row 4", "else branch of row 4"], [s["title"] for s in scopes])
        self.assertEqual([0, 0, 1, 1], [s["depth"] for s in scopes])

    def test_columns_say_where_each_version_is_live_and_released(self):
        function = {c["name"]: c for c in self.view("if-else")["scopes"][1]["columns"]}
        self.assertEqual("param", function["x#0"]["kind"])
        self.assertEqual("output", function["r#1"]["kind"])
        a = function["a#1"]
        self.assertEqual((0, 4, 5, 4), (a["defined"], a["live_end"], a["conservative_end"], a["release_row"]))
        # y#2, the condition register $t1#1 and a#1 are released at the same point, in that order
        self.assertEqual((1, 2, 3), (function["y#2"]["order"], function["$t1#1"]["order"], a["order"]))

    def test_a_branch_marks_what_leaves_it_as_returned(self):
        then_branch = {c["name"]: c for c in self.view("if-else")["scopes"][2]["columns"]}
        self.assertEqual("kept", then_branch["t#1"]["kind"])  # the release of y#2 needs it, so it outlives the branch
        self.assertEqual("output", then_branch["y#2"]["kind"])

    def test_every_released_version_in_the_view_is_released_in_the_output(self):
        # the view and the inserted code come from separate steps, so this checks they agree
        for p in PROGRAMS:
            with self.subTest(program=p["id"]):
                released = [line for line in text_of(compile_source(p["source"], FRONT_END[1:])).splitlines() if "~=" in line]
                for scope in self.view(p["id"])["scopes"]:
                    for column in scope["columns"]:
                        if column["kind"] == "temp":
                            self.assertTrue(any(f"{column['name']} ~=" in line for line in released), column["name"])


class AllocationViewTestCase(unittest.TestCase):
    def view(self, program_id, width=1, qubits=16):
        source = program(program_id)
        system = System(width, qubits)
        run = compiler(system).run(source, until="allocate")
        return allocation_view(run["lower_to_isa"], run["allocate"], system, source)

    def test_every_step_says_what_each_slot_held_and_where_it_came_from_in_the_source(self):
        steps = self.view("area")["steps"]
        inlined = [h for step in steps for h in step["held"] if h["register"].startswith("area1.p#")]
        self.assertTrue(inlined)
        self.assertEqual(("p", "area", [2]), (inlined[0]["variable"], inlined[0]["function"], inlined[0]["lines"]))  # `p = w * h;`
        caller = [h for step in steps for h in step["held"] if h["register"] == "x#1"]
        self.assertEqual(("x", "main"), (caller[0]["variable"], caller[0]["function"]))
        temporary = [h for step in steps for h in step["held"] if h["register"].startswith("$t")]
        self.assertTrue(not temporary or all(h["variable"] is None and h["lines"] == [] for h in temporary))

    def test_the_machine_and_what_the_program_needs_of_it(self):
        view = self.view("if-else", width=2, qubits=14)
        self.assertEqual((2, 14, 6, 5, True), (view["register_width"], view["qubits"], view["slots"], view["needed"], view["fits"]))
        self.assertEqual((2, 12), (view["scratch"], view["qubits_used"]))  # 5 registers of 2, and 2 for scratch

    def test_a_program_that_does_not_fit_says_so(self):
        from compiler.compiler import PassError
        system = System(4, 16)
        with self.assertRaises(PassError) as caught:
            compiler(system).run(program("if-else"), until="allocate")
        view = allocation_view(caught.exception.stages[-1].output, caught.exception.partial, system, program("if-else"))
        self.assertEqual((3, 5, False), (view["slots"], view["needed"], view["fits"]))

    def test_no_step_holds_two_registers_in_one_slot_and_the_steps_follow_the_program(self):
        for p in PROGRAMS:
            view = self.view(p["id"], width=1, qubits=1000)
            for step in view["steps"]:
                slots = [h["slot"] for h in step["held"]]
                self.assertEqual(len(slots), len(set(slots)))
            self.assertTrue(view["steps"][0]["logical"].startswith("ALLOC"))

    def test_a_registers_origin_is_read_from_its_name(self):
        self.assertEqual(("x", "main"), source_origin("x#2"))
        self.assertEqual(("p", "area"), source_origin("area1.p#1"))
        self.assertEqual(("m", "g"), source_origin("h1.g2.m#1"))
        self.assertEqual(("t", "main"), source_origin("t@3#1"))
        self.assertEqual((None, "main"), source_origin("$t4#1"))


if __name__ == '__main__':
    unittest.main()

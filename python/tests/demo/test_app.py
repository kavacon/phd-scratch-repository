import unittest

from demo import app
from demo.model import (
    Theme, Card, Case, Code, Columns, Diff, Metrics, Note, Number, Option, Result, Stage, Step, Switcher, Text, to_data, walk,
)


def toy_run(inputs, steps):
    if inputs["width"] == 13:
        raise ValueError("unlucky")
    return Result((Stage("only", "Only", blocks=(Note(f"{inputs['text']} {inputs['width']} {steps}"),)),))


DEMO = app.Demo(
    "Toy", "A demo of nothing", toy_run,
    inputs=(Text("text", "Text", default="hello"), Number("width", "Width", 8, 1, 16)),
    cases=(Case("a", "A", "the first", {"text": "from case"}),),
    steps=(Step("one", "One"), Step("two", "Two", requires=("one",))),
    actions={"more": lambda inputs, steps, payload: [Note(f"more {payload}")]},
)


class DemoTestCase(unittest.TestCase):
    def test_inputs_are_filled_in_and_kept_in_range(self):
        self.assertEqual({"text": "hello", "width": 8, "case": None}, DEMO.clean_inputs(None))
        self.assertEqual(16, DEMO.clean_inputs({"width": 99})["width"])
        self.assertEqual(8, DEMO.clean_inputs({"width": "x"})["width"])
        self.assertEqual("hello", DEMO.clean_inputs({"text": 5})["text"])

    def test_a_chosen_case_supplies_the_inputs_the_request_leaves_out(self):
        self.assertEqual(("from case", "a"), tuple(DEMO.clean_inputs({"case": "a"})[k] for k in ("text", "case")))
        self.assertEqual("typed", DEMO.clean_inputs({"case": "a", "text": "typed"})["text"])
        self.assertIsNone(DEMO.clean_inputs({"case": "nothing"})["case"])

    def test_steps_are_kept_in_the_demos_order_and_unknown_ones_dropped(self):
        self.assertEqual(["one", "two"], DEMO.clean_steps(["two", "nothing", "one"]))
        self.assertEqual(["one", "two"], DEMO.clean_steps(None))
        self.assertEqual([], DEMO.clean_steps([]))

    def test_a_demo_can_cap_how_many_steps_are_on_and_put_them_first(self):
        capped = app.Demo("t", "d", toy_run, steps=(Step("one", "One"), Step("two", "Two"), Step("three", "Three")),
                          steps_label="Pick", steps_first=True, max_steps=2)
        self.assertEqual(["one", "two"], capped.clean_steps(["three", "two", "one"]))
        self.assertEqual(["one", "two"], capped.clean_steps(None))
        boot = capped.bootstrap()
        self.assertEqual(("Pick", True, 2), (boot["steps_label"], boot["steps_first"], boot["max_steps"]))
        plain = DEMO.bootstrap()
        self.assertEqual(("Steps", False, 0), (plain["steps_label"], plain["steps_first"], plain["max_steps"]))

    def test_the_bootstrap_declares_inputs_cases_steps_and_action_urls(self):
        boot = DEMO.bootstrap()
        self.assertEqual(["text", "number"], [i["kind"] for i in boot["inputs"]])
        self.assertEqual("/api/action/more", boot["urls"]["actions"]["more"])
        self.assertEqual("/api/run", boot["urls"]["run"])
        self.assertEqual(("two", ["one"]), (boot["steps"][1]["id"], boot["steps"][1]["requires"]))

    def test_a_run_is_returned_as_data_and_a_failing_demo_becomes_an_error(self):
        data = DEMO.run_request({"inputs": {"text": "x", "width": 2}, "steps": ["one"]})
        self.assertEqual("note", data["stages"][0]["blocks"][0]["type"])
        self.assertIn("x 2 ['one']", data["stages"][0]["blocks"][0]["text"])
        self.assertIn("unlucky", DEMO.run_request({"inputs": {"width": 13}})["error"]["message"])

    def test_an_action_returns_blocks_and_a_failing_one_a_note(self):
        self.assertEqual("more {'k': 1}", DEMO.action_request("more", {"payload": {"k": 1}})["blocks"][0]["text"])
        failing = app.Demo("t", "d", toy_run, actions={"bad": lambda *_: 1 / 0})
        self.assertEqual("bad", failing.action_request("bad", {})["blocks"][0]["tone"])


class ThemeTestCase(unittest.TestCase):
    def test_a_demo_without_a_theme_keeps_the_shared_look(self):
        self.assertIsNone(DEMO.bootstrap()["theme"])

    def test_a_theme_goes_to_the_page_as_css_variables_per_scheme(self):
        demo = app.Demo("t", "d", toy_run, theme=Theme(accent="#0f766e", background="#f0fdfa", series_1="teal"))
        theme = demo.bootstrap()["theme"]
        self.assertEqual({"--accent": "#0f766e", "--bg": "#f0fdfa", "--s1": "teal"}, theme["light"])

    def test_surface_colours_stay_in_the_light_scheme_unless_the_dark_theme_gives_its_own(self):
        theme = Theme(accent="#0f766e", background="#f0fdfa", text="#042f2e").css()
        self.assertEqual({"--accent": "#0f766e"}, theme["dark"])  # brand colours carry over, a light background does not
        both = Theme(accent="#0f766e", background="#f0fdfa", dark=Theme(background="#042f2e", accent="#5eead4")).css()
        self.assertEqual({"--accent": "#5eead4", "--bg": "#042f2e"}, both["dark"])
        self.assertEqual({"--accent": "#0f766e", "--bg": "#f0fdfa"}, both["light"])

    def test_a_value_that_is_not_a_colour_is_refused(self):
        for bad in ("red; } body { display: none", "", "url(</style>)"):
            with self.assertRaises(ValueError):
                Theme(accent=bad)
        with self.assertRaises(ValueError):
            Theme(dark=Theme(dark=Theme()))


class ModelTestCase(unittest.TestCase):
    def test_blocks_carry_their_type_when_serialised(self):
        data = to_data(Switcher("v", (Option("a", "A", (Code.of("x"), Metrics((Card("l", "1"),)))),)))
        self.assertEqual("switcher", data["type"])
        self.assertEqual(["code", "metrics"], [b["type"] for b in data["options"][0]["blocks"]])

    def test_a_diff_marks_changes_and_says_how_many(self):
        diff = Diff.of("a = 1;\nb = 2;", "a = 1;\nb = 3;\nc = 4;", "Before", "After", key_of=lambda t: t[0] if "=" in t else None)
        self.assertEqual(["same", "changed", "added"], [line.kind for line in diff.after.lines])
        self.assertEqual("1 added · 1 changed", diff.after.meta.replace("1 changed · 1 added", "1 added · 1 changed"))
        self.assertEqual(["a", "b", "c"], [line.key for line in diff.after.lines])
        self.assertIsNone(Diff.of(None, "x").before)

    def test_walk_finds_blocks_nested_in_switchers_and_columns(self):
        tree = [Columns(((Note("a"),), (Switcher("s", (Option("o", "O", (Note("b"),)),)),)))]
        self.assertEqual(["columns", "note", "switcher", "note"], [to_data(b)["type"] for b in walk(tree)])


if __name__ == '__main__':
    unittest.main()

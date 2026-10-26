import unittest

from demo import app
from demo.model import Card, Case, Code, Diff, Metrics, Note, Number, Option, Result, Stage, Step, Switcher, Text, Theme, to_data


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
    def test_inputs_are_filled_in_kept_in_range_and_supplied_by_a_chosen_case(self):
        self.assertEqual({"text": "hello", "width": 8, "case": None}, DEMO.clean_inputs(None))
        self.assertEqual(16, DEMO.clean_inputs({"width": 99})["width"])
        self.assertEqual(("from case", "a"), tuple(DEMO.clean_inputs({"case": "a"})[k] for k in ("text", "case")))
        self.assertEqual("typed", DEMO.clean_inputs({"case": "a", "text": "typed"})["text"])

    def test_steps_keep_the_demos_order_and_a_demo_can_cap_how_many_are_on(self):
        self.assertEqual(["one", "two"], DEMO.clean_steps(["two", "nothing", "one"]))
        capped = app.Demo("t", "d", toy_run, steps=(Step("one", "One"), Step("two", "Two"), Step("three", "Three")),
                          steps_label="Pick", steps_first=True, max_steps=2)
        self.assertEqual(["one", "two"], capped.clean_steps(["three", "two", "one"]))
        self.assertEqual(("Pick", True, 2), tuple(capped.bootstrap()[k] for k in ("steps_label", "steps_first", "max_steps")))

    def test_the_bootstrap_declares_inputs_cases_steps_and_action_urls(self):
        boot = DEMO.bootstrap()
        self.assertEqual(["text", "number"], [i["kind"] for i in boot["inputs"]])
        self.assertEqual("/api/action/more", boot["urls"]["actions"]["more"])
        self.assertEqual(("two", ["one"]), (boot["steps"][1]["id"], boot["steps"][1]["requires"]))

    def test_a_run_is_returned_as_data_and_a_failing_demo_becomes_an_error(self):
        data = DEMO.run_request({"inputs": {"text": "x", "width": 2}, "steps": ["one"]})
        self.assertIn("x 2 ['one']", data["stages"][0]["blocks"][0]["text"])
        self.assertIn("unlucky", DEMO.run_request({"inputs": {"width": 13}})["error"]["message"])

    def test_an_action_returns_blocks(self):
        self.assertEqual("more {'k': 1}", DEMO.action_request("more", {"payload": {"k": 1}})["blocks"][0]["text"])


class ThemeTestCase(unittest.TestCase):
    def test_a_theme_goes_to_the_page_as_css_variables_and_surface_colours_stay_in_the_light_scheme(self):
        theme = app.Demo("t", "d", toy_run, theme=Theme(accent="#0f766e", background="#f0fdfa")).bootstrap()["theme"]
        self.assertEqual({"--accent": "#0f766e", "--bg": "#f0fdfa"}, theme["light"])
        self.assertEqual({"--accent": "#0f766e"}, theme["dark"])
        self.assertIsNone(DEMO.bootstrap()["theme"])

    def test_a_value_that_is_not_a_colour_is_refused(self):
        with self.assertRaises(ValueError):
            Theme(accent="red; } body { display: none")


class ModelTestCase(unittest.TestCase):
    def test_blocks_carry_their_type_when_serialised(self):
        data = to_data(Switcher("v", (Option("a", "A", (Code.of("x"), Metrics((Card("l", "1"),)))),)))
        self.assertEqual("switcher", data["type"])
        self.assertEqual(["code", "metrics"], [b["type"] for b in data["options"][0]["blocks"]])

    def test_a_diff_marks_what_changed(self):
        diff = Diff.of("a = 1;\nb = 2;", "a = 1;\nb = 3;\nc = 4;", "Before", "After", key_of=lambda t: t[0] if "=" in t else None)
        self.assertEqual(["same", "changed", "added"], [line.kind for line in diff.after.lines])


if __name__ == '__main__':
    unittest.main()

import unittest

from demo.text_diff import diff_lines


class DiffTestCase(unittest.TestCase):
    def kinds(self, before, after):
        old, new = diff_lines(before, after)
        return [entry["kind"] for entry in old], [entry["kind"] for entry in new]

    def test_unchanged_added_and_removed_lines(self):
        self.assertEqual((["same", "same"], ["same", "added", "same"]), self.kinds(["a", "c"], ["a", "b", "c"]))
        self.assertEqual((["same", "removed", "same"], ["same", "same"]), self.kinds(["a", "b", "c"], ["a", "c"]))

    def test_a_modified_line_highlights_only_what_changed(self):
        old, new = diff_lines(["    x = x + 1;"], ["    x#2 = x#1 + 1;"])
        self.assertEqual("changed", new[0]["kind"])
        text = new[0]["text"]
        self.assertEqual(["x#2", "x#1"], [text[s:e] for s, e in new[0]["spans"]])
        self.assertEqual(["x", "x"], [old[0]["text"][s:e] for s, e in old[0]["spans"]])

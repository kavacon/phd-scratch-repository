import unittest

from checking.simulate import simulate_isa
from compiler.parser.grammar import If
from compiler.parser.ast_tools import walk
from compiler.printer import unparse
from experiments.distributions import experiment

PREFIX = "a = 0..2;\nb = 0..2;\nx = 0;\ny = 0;\n"
NO_ELSE = PREFIX + "if a {\n    if b {\n        x = 3;\n    }\n}\nreturn x;\n"
OVERWRITES = PREFIX + "if a {\n    if b {\n        x = 3;\n    } else {\n        x = 1;\n    }\n}\nreturn x;\n"
NOT_OVERWRITTEN = PREFIX + "if a {\n    if b {\n        y = 1;\n    } else {\n        x = 1;\n    }\n}\nreturn x + y;\n"
OUTER_ELSE = PREFIX + "if a {\n    if b {\n        x = 3;\n    } else {\n        x = 1;\n    }\n} else {\n        x = 2;\n}\nreturn x;\n"
EXTRA_STATEMENT = PREFIX + "if a {\n    y = 2;\n    if b {\n        x = 3;\n    }\n}\nreturn x + y;\n"


def at(source, name):
    return experiment.compiler().run(source, until=name).output


def conditionals(nodes):
    return [n for s in nodes for n in walk(s) if isinstance(n, If)]


def depth(node):
    return 1 + max((depth(n) for s in node.body + node.orelse for n in walk(s) if isinstance(n, If)), default=0)


def result(source, flatten):
    skip = [] if flatten else ["flatten_conditionals", "flatten_controls"]
    program = experiment.compiler().run(source, until="release_values", skip=skip).output
    return simulate_isa(program, experiment.SYSTEM).value


class FlattenConditionalsTestCase(unittest.TestCase):
    def test_a_conditional_inside_a_conditional_with_no_else_becomes_one_conditional_on_the_conjunction(self):
        flat = at(NO_ELSE, "flatten_conditionals")
        self.assertEqual(1, len(conditionals(flat)))
        self.assertIn("$c1 = a && b;", unparse(flat))
        self.assertEqual(result(NO_ELSE, False), result(NO_ELSE, True))

    def test_the_inner_else_is_written_first_and_overwritten_when_the_inner_arm_assigns_everything_it_does(self):
        flat = at(OVERWRITES, "flatten_conditionals")
        self.assertEqual([1, 1], [depth(c) for c in conditionals(flat)])
        self.assertNotIn("!b", unparse(flat))  # no `a && !b` is needed
        self.assertEqual(result(OVERWRITES, False), result(OVERWRITES, True))

    def test_the_inner_else_keeps_its_own_condition_when_the_inner_arm_does_not_overwrite_what_it_writes(self):
        flat = at(NOT_OVERWRITTEN, "flatten_conditionals")
        self.assertIn("!b", unparse(flat))
        self.assertTrue(all(depth(c) == 1 for c in conditionals(flat)))
        self.assertEqual(result(NOT_OVERWRITTEN, False), result(NOT_OVERWRITTEN, True))

    def test_an_outer_else_is_kept_as_a_conditional_on_the_negation_and_other_nests_are_left_alone(self):
        flat = at(OUTER_ELSE, "flatten_conditionals")
        self.assertTrue(all(depth(c) == 1 for c in conditionals(flat)))
        self.assertEqual(result(OUTER_ELSE, False), result(OUTER_ELSE, True))
        self.assertEqual(2, max(depth(c) for c in conditionals(at(EXTRA_STATEMENT, "flatten_conditionals"))))
        self.assertEqual(result(EXTRA_STATEMENT, False), result(EXTRA_STATEMENT, True))


if __name__ == '__main__':
    unittest.main()

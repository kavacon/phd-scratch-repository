import unittest

from checking.simulate import simulate_isa
from compiler.parser.instructions import IsaProgram
from experiments.distributions import experiment
from experiments.distributions.passes.entanglement import analyse
from experiments.distributions.programs import PROGRAMS
from experiments.distributions.views import entangled_groups, groups_by_joint_operation

MIXED = next(p["source"] for p in PROGRAMS if p["id"] == "mixed")
UNOPTIMISED = ["collapse_updates", "fold_constants", "drop_unread", "fuse_steps", "drop_unread_versions"]
CONDITIONAL = "a = 0..2;\nb = 3;\nif a {\n    b = 5;\n}\nreturn b;\n"
SHARED = "a = 0..3;\np = 0..2;\nk = 2;\nx = a + k;\ny = p + k;\nreturn x;\n"
XOR_LIKE = "a = 0..2;\nb = a + 1;\nc = a + b;\nreturn c;\n"
TWO_IDENTITIES = "a = 0..2;\nb = 0..2;\nx = 1;\nif a {\n    if b {\n        x = 3;\n    }\n}\nreturn x;\n"


def isa(source, skip=(), until="lower_to_isa") -> IsaProgram:
    return experiment.compiler().run(source, until=until, skip=list(skip)).output


def names(groups):
    return sorted(sorted(g) for g in groups)


class AnalysisTestCase(unittest.TestCase):
    def test_a_result_is_joined_only_to_its_superposed_operands(self):
        program = isa(MIXED, UNOPTIMISED)
        final = analyse(program)[-1]
        self.assertEqual([["a#1", "b#1", "d#1"], ["c#1"], ["k#1"], ["m#1"]], names(final.groups))
        self.assertEqual(1, len(groups_by_joint_operation(program)))  # the POC's rule joins all six
        self.assertEqual({"a#1", "b#1", "d#1"}, final.superposed)

    def test_what_a_branch_writes_under_a_superposed_condition_is_superposed_and_joined_to_it(self):
        final = analyse(isa(CONDITIONAL, until="drop_noops"))[-1]
        self.assertIn("b#1", final.superposed)
        self.assertEqual([["a#1", "b#1"]], names(final.groups))
        copied = analyse(isa(CONDITIONAL))[-1]  # before the identity is aliased: b#2 is a copy of the constant b#1
        self.assertEqual([["a#1", "b#2"], ["b#1"]], names(copied.groups))

    def test_it_never_reports_less_entanglement_than_the_state_shows(self):
        programs = [p["source"] for p in PROGRAMS] + [CONDITIONAL, SHARED, XOR_LIKE, TWO_IDENTITIES, "a = 0..4;\nc = a * 3;\nreturn c;\n"]
        for source in programs:
            for skip in ([], UNOPTIMISED):
                for until in ("lower_to_isa", "release_values"):
                    program = isa(source, skip, until)
                    with self.subTest(source=source, skip=bool(skip), until=until):
                        held, state = simulate_isa(program, experiment.SYSTEM).state
                        final = analyse(program)[-1]
                        reported = [set(g) for g in final.groups]
                        for group in entangled_groups(held, state):
                            self.assertTrue(any(set(group) <= r for r in reported), group)
                        for k, name in enumerate(held):
                            if len({key[k] for key in state}) > 1:
                                self.assertIn(name, final.superposed)


if __name__ == '__main__':
    unittest.main()

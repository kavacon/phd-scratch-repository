import unittest

from compiler.passes.balance import balance
from experiments.allocation_poc.passes.liveness import analyse_program
from compiler.passes.rename import rename
from experiments.allocation_poc.experiment import compile_source


def parse(source):
    """The program as liveness sees it: balanced and renamed, before any releases are inserted."""
    return compile_source(source, passes=(balance, rename))


def dies_at(source, function, entanglement=True):
    return analyse_program(parse(source), entanglement)[function].dies_at


# Point k is immediately before statement k; the last point is the end of the block.
class LivenessTestCase(unittest.TestCase):
    AREA = """
        function area(w: int, h: int): int {
            p = w * h;
            q = p + 1;
            return q;
        }
        return area(2, 3);
    """

    def test_direct_uses_only(self):
        # w and h are last used by p's definition (statement 0), p by q's (statement 1)
        self.assertEqual({"w#0": 1, "h#0": 1, "p#1": 2}, dies_at(self.AREA, "area", entanglement=False))

    def test_entanglement_keeps_everything_connected_to_the_output_live(self):
        # q is computed from p, which was computed from w and h, so all stay live until the block ends (point 3)
        self.assertEqual({"w#0": 3, "h#0": 3, "p#1": 3}, dies_at(self.AREA, "area"))

    def test_variables_not_connected_to_the_output_die_early(self):
        source = """
            function g(x: int): int { t = 3; u = t + 4; r = x + 1; return r; }
            return g(1);
        """
        # t and u only touch each other and the constant 3, so they are dead from point 2 under both policies;
        # x is entangled with the output r, so it lives one point longer once entanglement is counted
        self.assertEqual({"x#0": 3, "t#1": 2, "u#1": 2}, dies_at(source, "g", entanglement=False))
        self.assertEqual({"x#0": 4, "t#1": 2, "u#1": 2}, dies_at(source, "g"))


def reclaim(source, function):
    analysis = analyse_program(parse(source))[function]
    return analysis.reclaim_at, analysis.release_order


class ReclamationTestCase(unittest.TestCase):
    """Where each temporary can be released once isolability is taken into account."""

    def test_temporary_is_released_right_after_its_last_use(self):
        # p feeds q, which is returned. Liveness keeps p to the end (dies_at 3) but it can be released at point 2
        source = "function area(w: int, h: int): int { p = w * h; q = p + 1; return q; } return area(2, 3);"
        self.assertEqual(({"p#1": 2}, ("p#1",)), reclaim(source, "area"))
        self.assertEqual(3, dies_at(source, "area")["p#1"])

    def test_versions_that_read_each_other_are_released_in_reverse_order(self):
        # b's release reads a, so b goes first even though both are last used at the same statement
        source = "function f(x: int): int { a = x + 1; b = a * 2; c = b + a; return c; } return f(1);"
        self.assertEqual(({"a#1": 3, "b#1": 3}, ("b#1", "a#1")), reclaim(source, "f"))

    def test_running_example_from_the_paper(self):
        # Shape of Faro et al.'s running example (Fig. 1, Fig. 3). t1 and t2 only feed y1, so they are released
        # right after y1 is computed (point 3). t3 and t4 feed y2, whose release reads them, so they wait for it.
        source = """
            function compute(x1: int, x2: int): int {
                t1 = f(x1);
                t2 = g(x1, t1);
                y1 = t1 + t2;
                t3 = h(x1);
                t4 = h(x2);
                y2 = t3 + t4;
                y3 = k(y2);
                return y1 + y3;
            }
            return compute(1, 2);
        """
        reclaim_at, order = reclaim(source, "compute")
        self.assertEqual({"t1#1": 3, "t2#1": 3, "t3#1": 7, "t4#1": 7, "y2#1": 7}, reclaim_at)
        self.assertEqual(("t2#1", "t1#1", "y2#1", "t4#1", "t3#1"), order)
        # entanglement-based liveness alone would keep every one of them until the end of the function (point 8)
        self.assertTrue(all(p == 8 for p in dies_at(source, "compute").values()))


def analysis(source, function):
    return analyse_program(parse(source))[function]


class ControlFlowTestCase(unittest.TestCase):
    """Blocks inside an `if` are analysed on their own, and the statement as a whole in its parent."""

    IF_ELSE = """
        function f(x: int): int {
            a = x + 1;
            y = 0;
            if a < 5 { t = a * 2; y = t + 1; } else { y = a - 1; }
            r = y + 1;
            return r;
        }
        return f(1);
    """

    def test_outer_version_used_in_a_branch_waits_for_the_whole_statement(self):
        # y#1 is overwritten in both branches so it is dead on arrival (point 2). a#1 is read inside the if, and
        # the version the if produces (y#2) reads everything the if read, so a#1 is only released after it (point 4).
        outer = analysis(self.IF_ELSE, "f")
        self.assertEqual({"y#1": 2, "y#2": 4, "a#1": 4}, outer.reclaim_at)
        self.assertEqual(("y#1", "y#2", "a#1"), outer.release_order)

    def test_a_branch_temporary_the_result_is_built_from_is_kept_for_the_release_of_the_if(self):
        # y#2 is built from t#1 in the then branch, and releasing y#2 later re-runs that recipe backwards, so t#1
        # cannot be released at the end of the branch. The parent's release of the if undoes both, latest first.
        then_block, else_block = analysis(self.IF_ELSE, "f").nested[2]
        self.assertEqual({}, then_block.reclaim_at)
        self.assertEqual((frozenset({"t#1"}), frozenset({"y#2", "t#1"})), (then_block.kept, then_block.release_later))
        self.assertEqual(frozenset({"y#2", "t#1"}), then_block.outputs)
        self.assertEqual((frozenset(), frozenset({"y#2"})), (else_block.kept, else_block.release_later))

    def test_a_branch_temporary_the_result_does_not_read_is_released_in_the_branch(self):
        source = """
            function f(a: int): int {
                x = 0;
                if a < 5 { u = a * 3; x = a + 1; } else { x = a - 1; }
                r = x + 1;
                return r;
            }
            return f(1);
        """
        then_block, _ = analysis(source, "f").nested[1]
        self.assertEqual(({"u#1": 1}, frozenset()), (then_block.reclaim_at, then_block.kept))

    def test_one_armed_if_is_balanced_so_the_old_version_is_released_after_it(self):
        # the missing else gets a copy of x#1, so x#1 is leftover on both paths and is released unconditionally
        source = "function f(a: int): int { x = a + 1; if x < 3 { x = x + 5; } return x; } return f(1);"
        outer = analysis(source, "f")
        self.assertEqual({"x#1": 2}, outer.reclaim_at)
        then_block, else_block = outer.nested[1]
        self.assertEqual(({}, frozenset({"x#2"})), (then_block.reclaim_at, then_block.outputs))
        self.assertEqual(({}, frozenset({"x#2"})), (else_block.reclaim_at, else_block.outputs))


if __name__ == '__main__':
    unittest.main()

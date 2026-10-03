import unittest

from parser.parser import parse
from passes.liveness import analyse_program


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

    def test_unused_values_are_dead_on_arrival(self):
        source = "function f(a: int, b: int): int { t = a + 1; return a; } return f(1, 2);"
        # a is returned so it never dies; b is never used; t is never used after it is defined
        self.assertEqual({"b#0": 0, "t#1": 1}, dies_at(source, "f", entanglement=False))

    def test_explicit_unassign_releases_a_version_and_its_entanglement(self):
        source = """
            function f(a: int): int {
                t = a + 1;
                u = t + 1;
                u ~= t + 1;
                t ~= a + 1;
                return a;
            }
            return f(1);
        """
        for entanglement in (False, True):
            self.assertEqual({"t#1": 4, "u#1": 3}, dies_at(source, "f", entanglement))

    def test_releasing_a_version_keeps_its_neighbours_entangled(self):
        # b was computed from x, which was computed from a, so b stays correlated with a after x is released;
        # a must therefore stay live while b does, even though a has no further direct use
        source = """
            function f(a: int): int {
                x = a + 1;
                b = x + 1;
                x ~= a + 1;
                return b;
            }
            return f(1);
        """
        self.assertEqual({"a#0": 4, "x#1": 3}, dies_at(source, "f"))
        self.assertEqual({"a#0": 3, "x#1": 3}, dies_at(source, "f", entanglement=False))

    def test_operands_stay_live_while_a_temporary_computed_from_them_is(self):
        # the paper's liveness: w and h are entangled with p, so neither can be released before p
        analysis = analyse_program(parse(self.AREA))["area"]
        for point, live in enumerate(analysis.live[:3]):
            self.assertLessEqual({"w#0", "h#0"}, live, f"point {point}")

    def test_outputs_never_die(self):
        analysis = analyse_program(parse(self.AREA))["area"]
        self.assertEqual(frozenset({"q#1"}), analysis.outputs)
        self.assertNotIn("q#1", analysis.dies_at)

    def test_top_level_program_and_functions_are_analysed_separately(self):
        analysis = analyse_program(parse(self.AREA))
        self.assertEqual({"<program>", "area"}, set(analysis))


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

    def test_unused_temporary_is_released_immediately(self):
        source = "function f(a: int): int { t = a + 1; return a; } return f(1);"
        self.assertEqual(({"t#1": 1}, ("t#1",)), reclaim(source, "f"))

    def test_parameters_and_outputs_are_never_released(self):
        source = "function f(a: int): int { t = a + 1; r = t + 1; return r; } return f(1);"
        reclaim_at, _ = reclaim(source, "f")
        self.assertEqual({"t#1"}, set(reclaim_at))

    def test_explicit_unassign_keeps_its_point_and_orders_its_operands(self):
        source = """
            function f(a: int): int {
                t = a + 1;
                u = t + 1;
                u ~= t + 1;
                t ~= a + 1;
                return a;
            }
            return f(1);
        """
        self.assertEqual(({"u#1": 3, "t#1": 4}, ("u#1", "t#1")), reclaim(source, "f"))

    def test_automatic_release_waits_for_an_explicit_one_that_reads_it(self):
        # u is released explicitly at point 4, and its witness reads t, so t cannot be released before then
        source = """
            function f(a: int): int {
                t = a + 1;
                u = t + 1;
                v = a + 2;
                u ~= t + 1;
                return v;
            }
            return f(1);
        """
        reclaim_at, order = reclaim(source, "f")
        self.assertEqual({"t#1": 4, "u#1": 4}, reclaim_at)
        self.assertEqual(("u#1", "t#1"), order)


def analysis(source, function):
    return analyse_program(parse(source))[function]


class ControlFlowTestCase(unittest.TestCase):
    """Blocks inside `if` and loops are analysed on their own, and the statement as a whole in its parent."""

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

    def test_branch_temporaries_are_released_by_the_end_of_their_block(self):
        then_block, else_block = analysis(self.IF_ELSE, "f").nested[2]
        self.assertEqual({"t#1": 2}, then_block.reclaim_at)  # t is local to the branch; y#2 is produced by both branches
        self.assertEqual(frozenset({"y#2"}), then_block.outputs)
        self.assertEqual({}, else_block.reclaim_at)
        self.assertEqual(frozenset({"y#2"}), else_block.outputs)

    def test_one_armed_if_is_balanced_so_the_old_version_is_released_after_it(self):
        # the missing else gets a copy of x#1, so x#1 is leftover on both paths and is released unconditionally
        source = "function f(a: int): int { x = a + 1; if x < 3 { x = x + 5; } return x; } return f(1);"
        outer = analysis(source, "f")
        self.assertEqual({"x#1": 2}, outer.reclaim_at)
        then_block, else_block = outer.nested[1]
        self.assertEqual(({}, frozenset({"x#2"})), (then_block.reclaim_at, then_block.outputs))
        self.assertEqual(({}, frozenset({"x#2"})), (else_block.reclaim_at, else_block.outputs))

    def test_loop_entry_values_are_merged_into_the_loop_target(self):
        source = "function g(n: int): int { s = 0; for i in 0 .. n { s = s + i; } return s; } return g(3);"
        self.assertEqual({}, analysis(source, "g").reclaim_at)  # s#1 enters the loop's phi, so it is not released alone

    def test_for_loop_body_temporaries_are_released_each_iteration(self):
        source = "function g(n: int): int { s = 0; for i in 0 .. n { t = s + i; s = t * 2; } return s; } return g(3);"
        outer = analysis(source, "g")
        self.assertEqual({}, outer.reclaim_at)  # s#1 enters the phi, s#2 is returned
        (body,) = outer.nested[1]
        self.assertEqual(({"t#1": 2}, frozenset({"s#3"})), (body.reclaim_at, body.outputs))

    def test_while_loop(self):
        source = "function f(n: int): int { c = 0; while c < n { d = c + 1; c = d; } return c; } return f(3);"
        outer = analysis(source, "f")
        self.assertEqual({}, outer.reclaim_at)
        (body,) = outer.nested[1]
        self.assertEqual({"d#1": 2}, body.reclaim_at)

    def test_loop_carried_value_that_is_not_returned_is_released_after_the_loop(self):
        source = """
            function h(n: int): int {
                a = 1; b = 2;
                for i in 0 .. n { if a < b { t = a + i; a = t; } b = b + a; }
                return a;
            }
            return h(3);
        """
        outer = analysis(source, "h")
        self.assertEqual({"b#2": 3}, outer.reclaim_at)  # b's header target is not returned
        (body,) = outer.nested[2]
        self.assertEqual({}, body.reclaim_at)
        then_block, _ = body.nested[0]  # the if inside the loop body
        self.assertEqual({"t#1": 2}, then_block.reclaim_at)

    def test_explicit_unassign_inside_a_block(self):
        source = "function f(a: int): int { r = a; if a < 3 { t = a + 1; t ~= a + 1; } return r; } return f(1);"
        (block, _) = analysis(source, "f").nested[1]
        self.assertEqual({"t#1": 2}, block.reclaim_at)

    def test_outer_versions_stay_live_across_the_statement_that_uses_them(self):
        outer = analysis(self.IF_ELSE, "f")
        self.assertIn("a#1", outer.live[2])  # before the if
        self.assertIn("a#1", outer.live[3])  # after it, still entangled with y#2


if __name__ == '__main__':
    unittest.main()

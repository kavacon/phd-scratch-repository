import unittest

from lark.exceptions import UnexpectedInput, VisitError

from compiler.parser.grammar import Assign, BinOp, Distribution, Number
from compiler.parser.parser import parse
from compiler.printer import unparse
from experiments.allocation_poc.programs import PROGRAMS


class DistributionSyntaxTestCase(unittest.TestCase):
    def test_a_range_assigned_to_a_name_is_a_distribution_in_grammar_v2_only(self):
        [statement, _] = parse("x = -2..3;\nreturn x;", "v2")
        self.assertEqual(Assign("x", Distribution(Number(-2), Number(3))), statement)
        self.assertEqual("x = -2..3;", unparse([statement]))
        with self.assertRaises(UnexpectedInput):
            parse("x = 0..4;\nreturn x;", "v1")

    def test_a_distribution_is_an_expression_with_whole_number_bounds_and_at_least_one_value(self):
        [statement, _] = parse("x = 0..4 + 1;\nreturn x;", "v2")
        self.assertEqual(Assign("x", BinOp("+", Distribution(Number(0), Number(4)), Number(1))), statement)
        with self.assertRaises(UnexpectedInput):
            parse("x = 0..1.5;\nreturn x;", "v2")
        for source in ("x = 3..3;", "x = 4..0;"):
            with self.assertRaises(VisitError, msg=source):
                parse(source + "\nreturn x;", "v2")

    def test_grammar_v2_reads_every_existing_program_as_v1_does(self):
        for program in PROGRAMS:
            self.assertEqual(parse(program["source"]), parse(program["source"], "v2"), program["id"])
        self.assertEqual(parse("x = 2.5 - -3;\nreturn x;"), parse("x = 2.5 - -3;\nreturn x;", "v2"))  # decimals and signs still read


if __name__ == '__main__':
    unittest.main()

import unittest

from lark.exceptions import UnexpectedInput

from compiler.parser.errors import describe_parse_error
from compiler.parser.parser import parse


def problem(source):
    try:
        parse(source)
    except UnexpectedInput as error:
        return describe_parse_error(error)
    raise AssertionError("the source parsed")


class ParseErrorTestCase(unittest.TestCase):
    def test_syntax_errors_are_reported_with_a_position(self):
        found = problem("function f(): int { return 1 + ; } return f();")
        self.assertEqual((1, "Unexpected ';'."), (found.line, found.message))

    def test_a_program_that_ends_early_says_every_program_ends_with_a_return(self):
        self.assertIn("return", problem("x = 1;").message)


if __name__ == '__main__':
    unittest.main()

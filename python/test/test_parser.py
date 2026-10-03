import unittest

from parser.parser import _parser

INPUT = """
x = 1;
y = x + 2;

function f(x: int): bool {
    y = x + 1;
    return y;
}

for x in 0 .. 10 {
    y = y + x;
}

while x == 1 {
    x = x - 1;
}

return y;
"""

class ParserTestCase(unittest.TestCase):
    def test_parse_ast(self):
        ast = _parser.parse(INPUT)
        variable_names = [x.value for x in ast.find_token("NAME")]

        self.assertIn("x", variable_names)
        self.assertIn("y", variable_names)
        self.assertIn("f", variable_names)
        self.assertEqual(1, len(list(ast.find_data("for_loop"))))
        self.assertEqual(1, len(list(ast.find_data("while_loop"))))
        self.assertEqual(1, len(list(ast.find_data("function_def"))))


if __name__ == '__main__':
    unittest.main()

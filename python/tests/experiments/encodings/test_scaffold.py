import io
import unittest
from contextlib import redirect_stdout

from compiler.compiler import CIRCUIT
from experiments.encodings import experiment
from experiments.encodings.encodings import DEFAULT, ENCODINGS


class EncodingsTestCase(unittest.TestCase):
    def test_every_encoding_has_a_lowering_that_is_a_circuit_pass(self):
        self.assertEqual(["binary", "binary-ttk", "fourier", "fourier-dual", "angle"], list(ENCODINGS))
        for encoding in ENCODINGS.values():
            self.assertEqual(CIRCUIT, encoding.lowering.compile_pass.gives, encoding.name)

    def test_an_encoding_not_built_yet_says_so_when_used(self):
        with self.assertRaises(NotImplementedError):
            experiment.run("return 1;", ["angle"])

    def test_the_command_line_prints_a_row_per_circuit_stage_of_each_default_encoding(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            experiment.main(["--program", "wrap"])
        lines = buffer.getvalue().strip().splitlines()
        self.assertEqual(1 + len(DEFAULT) * (1 + len(experiment.optimisers())), len(lines))  # a header, then each encoding's stages
        self.assertNotIn(" NO ", buffer.getvalue())


if __name__ == '__main__':
    unittest.main()

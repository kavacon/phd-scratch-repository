import io
import unittest
from contextlib import redirect_stdout
from dataclasses import fields

from compiler.compiler import CIRCUIT
from experiments.encodings import experiment
from experiments.encodings.encodings import DEFAULT, ENCODINGS, Encoding
from experiments.encodings.programs import PROGRAMS

PLACEHOLDERS = ["fourier-dual", "angle"]


class RegistryTestCase(unittest.TestCase):
    def test_the_encodings_in_the_plan(self):
        self.assertEqual(["binary", "binary-ttk", "fourier", "fourier-dual", "angle"], list(ENCODINGS))
        self.assertEqual(("binary", "binary-ttk", "fourier"), DEFAULT)

    def test_an_encoding_does_not_record_whether_it_is_ready(self):
        self.assertEqual(["name", "title", "summary", "lowering"], [f.name for f in fields(Encoding)])

    def test_every_lowering_is_a_circuit_pass_with_a_title_and_summary(self):
        for encoding in ENCODINGS.values():
            spec = encoding.lowering.compile_pass
            self.assertEqual(CIRCUIT, spec.gives, encoding.name)
            self.assertTrue(spec.title and spec.summary, encoding.name)

    def test_the_encodings_not_built_yet_say_so_when_used(self):
        for name in PLACEHOLDERS:
            with self.assertRaises(NotImplementedError, msg=name):
                ENCODINGS[name].lowering(None, None)

    def test_every_program_is_described(self):
        self.assertEqual(len(PROGRAMS), len({p["id"] for p in PROGRAMS}))
        for program in PROGRAMS:
            self.assertTrue(program["name"] and program["description"] and program["source"], program["id"])


class PipelineTestCase(unittest.TestCase):
    def test_every_encoding_loads_into_a_valid_pipeline_that_differs_only_in_its_lowering(self):
        names = {name: experiment.compiler(name).names for name in ENCODINGS}
        shared = names["binary"][:5]
        self.assertEqual(["inline", "unroll", "flatten", "balance", "rename"], shared)
        for name, passes in names.items():
            self.assertEqual(shared, passes[:5], name)
            self.assertEqual(["lower_to_isa", "allocate"], passes[5:7], name)
            self.assertEqual("optimise", passes[-1], name)
            self.assertEqual(ENCODINGS[name].lowering.compile_pass.name, passes[7], name)

    def test_there_is_no_release_insertion_in_the_pipeline(self):
        self.assertNotIn("lower_allocation", experiment.compiler("binary").names)

    def test_a_placeholder_encoding_fails_in_its_own_pass(self):
        with self.assertRaises(NotImplementedError):
            experiment.run("return 1;", ["angle"])


class CommandLineTestCase(unittest.TestCase):
    def output(self, *args):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            experiment.main(list(args))
        return buffer.getvalue()

    def test_list_shows_every_encoding(self):
        text = self.output("--list")
        for name in ENCODINGS:
            self.assertIn(name, text)
        self.assertNotIn("placeholder", text)

    def test_plan_shows_the_passes_of_the_chosen_encodings(self):
        text = self.output("--encoding", "binary", "fourier", "--plan")
        self.assertIn("binary: source -> inline", text)
        self.assertIn("to_circuit_fourier", text)

    def test_running_prints_a_row_per_circuit_stage_per_encoding(self):
        text = self.output("--program", "wrap")
        lines = text.strip().splitlines()
        self.assertEqual(1 + len(DEFAULT) * 2, len(lines))  # a header, then each encoding with two circuit stages
        for name in DEFAULT:
            self.assertIn(name, text)
        self.assertNotIn(" NO ", text)


if __name__ == '__main__':
    unittest.main()

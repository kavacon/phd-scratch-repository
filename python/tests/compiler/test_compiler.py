import unittest

from compiler.compiler import CIRCUIT, ISA, NODE, Compiler, PassError, compile_pass
from compiler.system import System


@compile_pass(NODE, NODE)
def double(program, system=None):
    return program + program


@compile_pass(NODE, NODE)
def fails(program, system=None):
    raise KeyError("boom")


@compile_pass(NODE, ISA)
def to_text(program, system=None):
    return f"{len(program)} nodes on {system.register_width}"


@compile_pass(ISA, ISA)
def shout(text, system=None):
    return text.upper()


@compile_pass(ISA, CIRCUIT)
def to_circuit(text, system=None):
    return [text]


class NamesTestCase(unittest.TestCase):
    def test_a_pass_is_named_by_its_function_and_titled_from_that_unless_told_otherwise(self):
        self.assertEqual(("to_text", "To text"), (to_text.compile_pass.name, to_text.compile_pass.title))


class SystemTestCase(unittest.TestCase):
    def test_slots_leave_one_register_for_scratch(self):
        self.assertEqual(14, System(register_width=6, qubits=90).slots)
        self.assertEqual(0, System(register_width=8, qubits=8).slots)


class LoadTestCase(unittest.TestCase):
    def test_passes_must_connect_level_to_level(self):
        Compiler().load(double, to_text, shout, to_circuit)
        with self.assertRaises(ValueError):
            Compiler().load(to_text, double)  # a node pass cannot follow the lowering to the ISA
        with self.assertRaises(ValueError):
            Compiler().load(shout)  # the first pass takes nodes


class RunTestCase(unittest.TestCase):
    SOURCE = "x = 1; return x;"

    def test_every_stage_is_kept_with_its_level_and_the_system_is_passed_on(self):
        result = Compiler(System(register_width=4)).load(double, to_text, shout, to_circuit).run(self.SOURCE)
        self.assertEqual(["source", "double", "to_text", "shout", "to_circuit"], result.names)
        self.assertEqual(["node", "node", "isa", "isa", "circuit"], [s.level for s in result.stages])
        self.assertEqual("4 NODES ON 4", result["shout"])
        self.assertEqual(["4 NODES ON 4"], result.circuit)
        self.assertEqual(2 * len(result["source"]), len(result["double"]))

    def test_it_can_stop_after_a_pass_and_leave_passes_out(self):
        compiler = Compiler().load(double, to_text, shout)
        self.assertEqual(["source", "double"], compiler.run(self.SOURCE, until="double").names)
        self.assertEqual(["source", "to_text", "shout"], compiler.run(self.SOURCE, skip=["double"]).names)
        with self.assertRaises(ValueError):
            compiler.run(self.SOURCE, until="nothing")

    def test_a_failing_pass_raises_with_everything_before_it(self):
        with self.assertRaises(PassError) as caught:
            Compiler().load(double, fails, to_text).run(self.SOURCE)
        error = caught.exception
        self.assertEqual("fails", error.name)
        self.assertEqual(["source", "double"], [s.name for s in error.stages])
        self.assertIsInstance(error.cause, KeyError)


if __name__ == '__main__':
    unittest.main()

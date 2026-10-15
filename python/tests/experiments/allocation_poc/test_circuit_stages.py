import unittest

from qiskit import QuantumCircuit

from checking import circuits, qiskit_tools
from checking.simulate import execute, simulate
from compiler.parser.parser import parse
from compiler.compiler import PassError
from compiler.passes.optimise import optimise
from compiler.passes.qasm import measured_qubits, openqasm_text, x_gates
from compiler.system import System
from experiments.allocation_poc.experiment import compile_circuit, compiler
from experiments.allocation_poc.programs import PROGRAMS

needs_qiskit = unittest.skipUnless(qiskit_tools.qiskit_available(), "Qiskit is not installed")
needs_aer = unittest.skipUnless(qiskit_tools.qiskit_available() and qiskit_tools.aer_available(), "Qiskit Aer is not installed")

SMALL = [
    "a = 3; b = a + 2; return b;",
    "a = 2; b = 3; c = a * b; return c;",
    "a = 2; b = 5; c = a < b; return c;",
    "a = 4; b = 4; c = a == b; return c;",
    "a = 5; y = 0; if a < 3 { y = a + 1; } else { y = a - 1; } return y;",
]


@needs_qiskit
class QiskitTestCase(unittest.TestCase):
    def test_the_printed_text_loads_into_qiskit_with_the_same_wires_and_gate_count_as_the_circuit(self):
        for source in SMALL + [p["source"] for p in PROGRAMS]:
            circuit = compile_circuit(source, System(6, 1000))
            loaded = qiskit_tools.read(openqasm_text(circuit))
            self.assertEqual(circuit.num_qubits, loaded.num_qubits)
            self.assertEqual(len(x_gates(circuit)), qiskit_tools.measure(loaded)["gates"], source[:40])

    def test_the_optimised_circuit_uses_only_u_and_cx(self):
        optimised = optimise(compile_circuit(SMALL[1], System(3, 1000)))
        self.assertLessEqual(set(qiskit_tools.measure(optimised)["ops"]), {"u", "cx", "measure"})

    def test_measurements_are_counted_apart_from_gates_and_survive_optimisation(self):
        circuit = compile_circuit("a = 3; b = a + 2; return b;", System(3, 1000))
        before, after = qiskit_tools.measure(circuit), qiskit_tools.measure(optimise(circuit))
        self.assertEqual((len(measured_qubits(circuit)), len(measured_qubits(circuit))), (before["measurements"], after["measurements"]))
        self.assertEqual(len(x_gates(circuit)), before["gates"])  # the measurements are not gates
        self.assertEqual(len(measured_qubits(circuit)), before["ops"]["measure"])

    def test_measure_counts_gates_depth_and_the_wires_that_are_touched(self):
        loaded = qiskit_tools.read("OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[5] q;\nx q[0];\ncx q[0], q[1];\nccx q[0], q[1], q[2];\n")
        measured = qiskit_tools.measure(loaded)
        self.assertEqual((3, 2, 3, 5, 3), (measured["gates"], measured["two_qubit"], measured["depth"], measured["wires"], measured["touched"]))
        self.assertEqual({"x": 1, "cx": 1, "ccx": 1}, measured["ops"])


@needs_qiskit
class CircuitPassesTestCase(unittest.TestCase):
    """The circuit steps are passes like the rest, and both give a Qiskit circuit."""

    def run_all(self, source, width=6, qubits=200):
        return {s.name: s.output for s in compiler(System(width, qubits)).run(source).stages}

    def test_the_code_generator_and_the_optimiser_are_the_last_two_passes(self):
        stages = self.run_all(PROGRAMS[0]["source"])
        self.assertIsInstance(stages["to_openqasm"], QuantumCircuit)
        self.assertIsInstance(stages["optimise"], QuantumCircuit)
        self.assertEqual(["to_openqasm", "optimise"], list(stages)[-2:])

    def test_both_circuits_have_the_same_wires_and_the_optimised_one_is_u_and_cx_only(self):
        stages = self.run_all(PROGRAMS[0]["source"])
        generated, optimised = qiskit_tools.measure(stages["to_openqasm"]), qiskit_tools.measure(stages["optimise"])
        self.assertEqual(generated["wires"], optimised["wires"])
        self.assertLessEqual(optimised["touched"], optimised["wires"])
        self.assertLessEqual(set(optimised["ops"]), {"u", "cx", "measure"})

    def test_a_circuit_has_no_reference_machine_so_it_is_only_run_on_request(self):
        stages = self.run_all(PROGRAMS[0]["source"])
        self.assertIsNone(execute(stages["to_openqasm"]))
        self.assertIsNone(execute(stages["optimise"]))

    @needs_aer
    def test_running_the_optimised_circuit_returns_what_the_program_returns_and_leaves_nothing_set(self):
        source = PROGRAMS[0]["source"]
        stages = self.run_all(source)
        outcome = qiskit_tools.execute(stages["optimise"], len(measured_qubits(stages["to_openqasm"])), 20)
        report = circuits.report(outcome, shots=20, expected=circuits.wrap(simulate(parse(source)).value, 6))
        self.assertEqual(([[13, 20]], True, []), (report["counts"], report["matches"], report["leaked"]))

    def test_a_division_program_stops_at_the_code_generator_with_an_explanation(self):
        with self.assertRaises(PassError) as caught:
            compiler(System(6, 100)).run("a = 8; b = a / 2; return b;")
        self.assertEqual("to_openqasm", caught.exception.name)
        self.assertIn("div is not supported", str(caught.exception))


if __name__ == '__main__':
    unittest.main()

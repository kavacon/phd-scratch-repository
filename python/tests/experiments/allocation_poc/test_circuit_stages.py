import unittest

from checking import circuits, qiskit_tools
from checking.simulate import simulate
from compiler.parser.parser import parse
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


@needs_qiskit
class CircuitPassesTestCase(unittest.TestCase):
    """The circuit steps are passes like the rest, and both give a Qiskit circuit."""

    def run_all(self, source, width=6, qubits=200):
        return {s.name: s.output for s in compiler(System(width, qubits)).run(source).stages}

    def test_both_circuits_have_the_same_wires_and_the_optimised_one_is_u_and_cx_only(self):
        stages = self.run_all(PROGRAMS[0]["source"])
        generated, optimised = qiskit_tools.measure(stages["to_openqasm"]), qiskit_tools.measure(stages["optimise"])
        self.assertEqual(generated["wires"], optimised["wires"])
        self.assertLessEqual(optimised["touched"], optimised["wires"])
        self.assertLessEqual(set(optimised["ops"]), {"u", "cx", "measure"})

    @needs_aer
    def test_running_the_optimised_circuit_returns_what_the_program_returns_and_leaves_nothing_set(self):
        source = PROGRAMS[0]["source"]
        stages = self.run_all(source)
        outcome = qiskit_tools.execute(stages["optimise"], len(measured_qubits(stages["to_openqasm"])), 20)
        report = circuits.report(outcome, shots=20, expected=circuits.wrap(simulate(parse(source)).value, 6))
        self.assertEqual(([[13, 20]], True, []), (report["counts"], report["matches"], report["leaked"]))


if __name__ == '__main__':
    unittest.main()

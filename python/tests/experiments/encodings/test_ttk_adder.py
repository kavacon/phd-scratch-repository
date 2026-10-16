import itertools
import unittest

from qiskit import QuantumCircuit

from checking.circuits import run_gates
from compiler.passes.qasm import x_gates
from experiments.encodings.passes.binary_ttk import _TtkBuilder


def adder(n, controls=(), extra=0):
    """The adder `b += a` on n-bit registers a (wires 0..n-1) and b (n..2n-1), plus `extra` spare wires after them."""
    circuit = QuantumCircuit(2 * n + extra)
    builder = _TtkBuilder(circuit, n, 2 * n, regions=controls)
    builder.adder(list(range(n, 2 * n)), list(range(n)), carry=2 * n)
    return circuit


def run(circuit, n, a, b, spare=()):
    state = [(a >> i) & 1 for i in range(n)] + [(b >> i) & 1 for i in range(n)] + list(spare)
    out = run_gates(x_gates(circuit), circuit.num_qubits, state)
    return sum(out[i] << i for i in range(n)), sum(out[n + i] << i for i in range(n)), out[2 * n:]


class TtkAdderTestCase(unittest.TestCase):
    def test_it_adds_modulo_2_to_the_n_and_restores_the_addend_for_every_input(self):
        for n in range(1, 7):
            circuit = adder(n)
            for a, b in itertools.product(range(2 ** n), repeat=2):
                self.assertEqual((a, (a + b) % 2 ** n), run(circuit, n, a, b)[:2], (n, a, b))

    def test_it_uses_no_ancilla_not_even_the_carry_qubit_it_is_offered(self):
        for n in range(1, 7):
            touched = {q for gate in x_gates(adder(n, extra=1)) for q in (gate.target, *[c for c, _ in gate.controls])}
            self.assertEqual(set(range(2 * n)), touched & set(range(2 * n + 1)) | touched, n)
            self.assertNotIn(2 * n, touched, n)

    def test_the_gate_count_is_the_papers_without_the_carry_out(self):
        for n in range(2, 8):
            gates = x_gates(adder(n))
            self.assertEqual(5 * n - 6, sum(1 for g in gates if len(g.controls) == 1), n)
            self.assertEqual(2 * n - 2, sum(1 for g in gates if len(g.controls) == 2), n)

    def test_controlled_on_a_qubit_it_adds_only_when_that_qubit_is_set(self):
        n = 3
        circuit = adder(n, controls=[(2 * n, True)], extra=1)
        for a, b in itertools.product(range(8), repeat=2):
            self.assertEqual(b, run(circuit, n, a, b, spare=[0])[1])
            self.assertEqual((a + b) % 8, run(circuit, n, a, b, spare=[1])[1])

    def test_its_inverse_subtracts(self):
        n = 3
        circuit = adder(n).inverse()
        for a, b in itertools.product(range(8), repeat=2):
            self.assertEqual((a, (b - a) % 8), run(circuit, n, a, b)[:2])


if __name__ == '__main__':
    unittest.main()

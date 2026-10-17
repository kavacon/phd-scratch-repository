import itertools
import unittest

from qiskit import QuantumCircuit

from checking.circuits import run_gates
from compiler.passes.qasm import x_gates
from experiments.encodings.passes.binary_ttk import _TtkBuilder


def adder(n, controls=(), extra=0):
    """The adder `b += a` on n-bit registers a (wires 0..n-1) and b (n..2n-1), plus `extra` spare wires after them."""
    circuit = QuantumCircuit(2 * n + extra)
    _TtkBuilder(circuit, n, 2 * n, regions=controls).adder(list(range(n, 2 * n)), list(range(n)), carry=2 * n)
    return circuit


def run(circuit, n, a, b, spare=()):
    state = [(a >> i) & 1 for i in range(n)] + [(b >> i) & 1 for i in range(n)] + list(spare)
    out = run_gates(x_gates(circuit), circuit.num_qubits, state)
    return sum(out[i] << i for i in range(n)), sum(out[n + i] << i for i in range(n))


class TtkAdderTestCase(unittest.TestCase):
    def test_it_adds_modulo_2_to_the_n_and_restores_the_addend(self):
        for n in range(1, 6):
            circuit = adder(n)
            for a, b in itertools.product(range(2 ** n), repeat=2):
                self.assertEqual((a, (a + b) % 2 ** n), run(circuit, n, a, b), (n, a, b))

    def test_controlled_on_a_qubit_it_adds_only_when_that_qubit_is_set(self):
        n = 3
        circuit = adder(n, controls=[(2 * n, True)], extra=1)
        for a, b in itertools.product(range(8), repeat=2):
            self.assertEqual(b, run(circuit, n, a, b, spare=[0])[1])
            self.assertEqual((a + b) % 8, run(circuit, n, a, b, spare=[1])[1])


if __name__ == '__main__':
    unittest.main()

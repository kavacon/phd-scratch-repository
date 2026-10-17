import itertools
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from qiskit.synthesis.qft import synth_qft_full

from experiments.encodings import experiment
from experiments.encodings.passes.fourier import _FourierBuilder, _transform
from experiments.encodings.programs import PROGRAMS
from compiler.system import System


def transform_circuit(n, inverse=False):
    circuit = QuantumCircuit(n)
    gates = _transform(list(range(n)))
    for gate in (reversed(gates) if inverse else gates):
        if gate[0] == "h":
            circuit.h(gate[1])
        else:
            circuit.cp(-gate[1] if inverse else gate[1], gate[2], gate[3])
    return circuit


class FourierTestCase(unittest.TestCase):
    def test_the_transform_is_qiskits_without_the_swaps_and_the_inverse_undoes_it(self):
        for n in range(1, 5):
            self.assertTrue(Operator(transform_circuit(n)).equiv(Operator(synth_qft_full(n, do_swaps=False))), n)
            both = transform_circuit(n).compose(transform_circuit(n, inverse=True))
            self.assertTrue(Operator(both).equiv(Operator(QuantumCircuit(n))), n)

    def test_a_constant_added_in_the_fourier_basis_wraps_modulo_two_to_the_n(self):
        n = 3
        for x, k in itertools.product(range(2 ** n), range(-2 ** n, 2 ** n + 1)):
            circuit = QuantumCircuit(n)
            builder = _FourierBuilder(circuit, n, n, [], set())
            for q in range(n):
                if (x >> q) & 1:
                    circuit.x(q)
            builder.to_fourier(list(range(n)))
            builder.rotate(list(range(n)), k)
            builder.to_plain(list(range(n)))
            best = int(np.argmax(Statevector(circuit).probabilities()))
            self.assertEqual((x + k) % 2 ** n, best, (x, k))

    def test_a_register_converts_only_when_something_needs_it_in_the_other_basis(self):
        def ops(source):
            return experiment.compile_all(source, ["fourier"], System(4, 100), zx=False)["fourier"]["to_circuit_fourier"].count_ops()
        one = ops("a = 3; c = a + 1; c += 2; return c;")
        many = ops("a = 3; c = a + 1; c += 2; c += 5; c += 1; return c;")
        compared = ops("a = 3; c = a + 1; c += 2; t = c < 4; c += 5; return c;")
        self.assertEqual(one["h"], many["h"])  # more additions, no more transforms
        self.assertGreater(compared["h"], one["h"])  # a comparison in between forces a conversion back

    def test_every_program_returns_what_the_reference_returns(self):
        for program in PROGRAMS:
            for row in experiment.run(program["source"], ["fourier"], System(4, 100), zx=False):
                self.assertTrue(row.agrees, (program["id"], row))


if __name__ == '__main__':
    unittest.main()

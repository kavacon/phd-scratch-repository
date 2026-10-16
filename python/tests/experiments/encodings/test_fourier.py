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


def result_of(circuit, n, qubits=None):
    state = Statevector(circuit)
    probabilities = state.probabilities(qubits or list(range(n)))
    best = int(np.argmax(probabilities))
    return best, probabilities[best]


class TransformTestCase(unittest.TestCase):
    def test_the_transform_is_qiskits_without_the_swaps(self):
        for n in range(1, 6):
            self.assertTrue(Operator(transform_circuit(n)).equiv(Operator(synth_qft_full(n, do_swaps=False))), n)

    def test_the_inverse_undoes_it(self):
        for n in range(1, 6):
            both = transform_circuit(n).compose(transform_circuit(n, inverse=True))
            self.assertTrue(Operator(both).equiv(Operator(QuantumCircuit(n))), n)

    def test_the_transform_of_zero_is_a_hadamard_on_each_qubit(self):
        for n in range(1, 6):
            hadamards = QuantumCircuit(n)
            hadamards.h(range(n))
            self.assertTrue(Statevector(transform_circuit(n)).equiv(Statevector(hadamards)), n)


class RotationTestCase(unittest.TestCase):
    """Adding into a register in the Fourier basis, then converting back, is addition modulo 2^n."""

    def builder(self, n, extra=0):
        circuit = QuantumCircuit(n + extra)
        return circuit, _FourierBuilder(circuit, n, n + extra, [], set())

    def run_add(self, n, x, k):
        circuit, builder = self.builder(n)
        for q in range(n):
            if (x >> q) & 1:
                circuit.x(q)
        builder.to_fourier(list(range(n)))
        builder.rotate(list(range(n)), k)
        builder.to_plain(list(range(n)))
        return result_of(circuit, n)

    def test_a_constant_added_into_a_register_wraps_modulo_two_to_the_n(self):
        for n in (1, 2, 3, 4):
            for x, k in itertools.product(range(2 ** n), range(-2 ** n, 2 ** n + 1)):
                value, certainty = self.run_add(n, x, k)
                self.assertEqual((x + k) % 2 ** n, value, (n, x, k))
                self.assertAlmostEqual(1.0, certainty)

    def test_a_register_added_by_its_bits_adds_only_what_each_set_bit_weighs(self):
        n = 3
        for x, a in itertools.product(range(8), repeat=2):
            circuit = QuantumCircuit(2 * n)
            builder = _FourierBuilder(circuit, n, 2 * n, [], set())
            for q in range(n):
                if (x >> q) & 1: circuit.x(q)
                if (a >> q) & 1: circuit.x(n + q)
            builder.to_fourier([0, 1, 2])
            for m in range(n):
                builder.rotate([0, 1, 2], 2 ** m, [(n + m, True)])
            builder.to_plain([0, 1, 2])
            self.assertEqual(((x + a) % 8, a), (result_of(circuit, n)[0], result_of(circuit, n, [3, 4, 5])[0]))

    def test_a_negative_value_subtracts(self):
        self.assertEqual(2, self.run_add(3, 5, -3)[0])
        self.assertEqual(7, self.run_add(3, 2, -3)[0])  # wraps


class TrackingTestCase(unittest.TestCase):
    def ops(self, source, width=4):
        compilation = experiment.compile_all(source, ["fourier"], System(width, 100))["fourier"]
        return dict(compilation["to_circuit_fourier"].count_ops())

    def test_consecutive_additions_into_one_register_convert_it_once(self):
        one = self.ops("a = 3; c = a + 1; c += 2; return c;")
        three = self.ops("a = 3; c = a + 1; c += 2; c += 5; c += 1; return c;")
        self.assertEqual(one["h"], three["h"])
        self.assertEqual(one["cp"], three["cp"])  # more rotations are plain phase gates, not more transforms
        self.assertGreater(three["p"], one["p"])

    def test_a_comparison_in_between_forces_the_register_back_to_plain_bits(self):
        plain = self.ops("a = 3; c = a + 1; c += 2; c += 5; return c;")
        compared = self.ops("a = 3; c = a + 1; c += 2; t = c < 4; c += 5; return c;")
        self.assertGreater(compared["h"], plain["h"])
        self.assertGreater(compared["cp"], plain["cp"])

    def test_a_program_that_compares_only_converts_nothing(self):
        ops = self.ops("a = 3; b = 5; t = a < b; return t;")
        self.assertNotIn("h", ops)
        self.assertNotIn("cp", ops)

    def test_adding_registers_needs_no_scratch_qubit(self):
        rows = experiment.run("a = 3; b = 2; c = a + b; d = c + 4; return d;", ["fourier"], System(4, 100))
        self.assertEqual(4 * 4, rows[0].qubits_touched)  # a, b, c and d: nothing borrowed from scratch

    def test_the_two_branches_of_an_if_else_may_write_the_same_register(self):
        source = "a = -3; b = 2; t = a < b; r = 0; if t { r = a + b; } else { r = a - b; } return r;"
        for width in (3, 4):
            for row in experiment.run(source, ["fourier"], System(width, 100)):
                self.assertTrue(row.agrees, row)


class AgreementTestCase(unittest.TestCase):
    def test_every_program_returns_what_the_reference_returns(self):
        for width in (3, 4, 5):
            for program in PROGRAMS:
                for row in experiment.run(program["source"], ["fourier"], System(width, 100)):
                    self.assertTrue(row.agrees, (width, program["id"], row))

    def test_the_circuit_is_measured_as_signed_bits_in_one_shot(self):
        compilation = experiment.compile_all("a = -3; return a;", ["fourier"], System(4, 100))["fourier"]
        self.assertEqual({"shots": 1, "decode": "signed-bits"}, compilation["to_circuit_fourier"].metadata["readout"])


if __name__ == '__main__':
    unittest.main()

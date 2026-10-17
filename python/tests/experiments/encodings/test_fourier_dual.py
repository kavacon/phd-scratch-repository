import unittest

from qiskit import QuantumCircuit

from compiler.passes.qasm import Q
from compiler.system import System
from experiments.encodings import experiment
from experiments.encodings.passes.fourier_dual import _DualBuilder
from experiments.encodings.programs import PROGRAMS


class FourierDualTestCase(unittest.TestCase):
    def test_adding_one_register_into_another_in_the_fourier_basis_is_only_cnot_and_toffoli_gates(self):
        n = 4
        circuit = QuantumCircuit(2 * n + 1)  # two registers and the carry qubit
        builder = _DualBuilder(circuit, n, 2 * n, [], {0, 1})  # both registers are already in the Fourier basis
        builder.update("add", list(range(n)), [Q(i) for i in range(n, 2 * n)])
        self.assertEqual({"cx", "ccx"}, set(circuit.count_ops()))

    def test_every_program_returns_what_the_reference_returns(self):
        for program in PROGRAMS:
            for row in experiment.run(program["source"], ["fourier-dual"], System(4, 100), zx=False):
                self.assertTrue(row.agrees, (program["id"], row))


if __name__ == '__main__':
    unittest.main()

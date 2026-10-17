import unittest

from qiskit import QuantumCircuit

from compiler.passes.qasm import measured_qubits
from compiler.system import System
from experiments.encodings import experiment
from experiments.encodings.passes.optimise_zx import zx_available

SYSTEM = System(3, 100)
SOURCE = "a = 1; b = 2; c = a + b; c += 1; t = c < 3; return c;"


@unittest.skipUnless(zx_available(), "PyZX is not installed")
class OptimiseZxTestCase(unittest.TestCase):
    def test_the_zx_circuit_follows_qiskits_and_reads_out_what_the_reference_returns(self):
        for name in ("binary-ttk", "fourier"):
            rows = experiment.run(SOURCE, [name], SYSTEM)
            self.assertEqual(["optimise", "optimise_zx"], [r.stage for r in rows[1:]])
            self.assertTrue(all(r.agrees for r in rows), rows)

    def test_the_measurements_and_readout_recipe_survive(self):
        compilation = experiment.compile_all(SOURCE, ["fourier"], SYSTEM)["fourier"]
        lowered, zx = compilation.stages[-3].output, compilation["optimise_zx"]
        self.assertEqual(measured_qubits(lowered), measured_qubits(zx))
        self.assertEqual(lowered.metadata, zx.metadata)


class MetricsTestCase(unittest.TestCase):
    def test_non_clifford_counts_t_like_gates_and_two_qubit_depth_only_the_multi_qubit_ones(self):
        toffoli = QuantumCircuit(3)
        toffoli.ccx(0, 1, 2)
        clifford = QuantumCircuit(2)
        clifford.h(0); clifford.s(1); clifford.cx(0, 1)
        self.assertEqual((7, 0), (experiment.non_clifford(toffoli), experiment.non_clifford(clifford)))
        layered = QuantumCircuit(4)
        layered.cx(0, 1); layered.cx(2, 3); layered.h(0); layered.cx(1, 2)
        self.assertEqual(2, experiment.count(layered)["two_qubit_depth"])


if __name__ == '__main__':
    unittest.main()

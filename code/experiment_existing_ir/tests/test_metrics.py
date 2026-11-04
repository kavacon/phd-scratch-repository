import subprocess
import unittest

from compiler.parser.parser import parse
from compiler.system import System
from experiment_existing_ir import metrics, pipeline
from experiment_existing_ir.translate import translate


def build(source: str, *stages: str):
    return pipeline.build(translate(parse(source, "v2")), *(stages or pipeline.DEFAULT))


@unittest.skipUnless(pipeline.built(), "existing-ir-opt is not built: run helper/cpp/setup.sh and cpp/build.sh")
class Metrics(unittest.TestCase):
    def test_a_circuit_is_counted_with_its_adders_written_out_and_t_cost_taken_from_toffolis(self):
        circuit = metrics.of_qc(build("a = 0..4; b = 0..3; return a * b;"))
        self.assertGreater(circuit.toffolis, 0)
        self.assertEqual(circuit.t_count, metrics.TOFFOLI_T_COUNT * circuit.toffolis)
        self.assertGreater(circuit.gates, circuit.toffolis)  # the adders' gates are counted, not the calls
        # the same circuit is measured whether it is given as QC or as the OpenQASM 3 written from it
        self.assertEqual(circuit, metrics.of_openqasm(pipeline.export(build("a = 0..4; b = 0..3; return a * b;"), "openqasm3")))

    def test_an_interval_that_is_not_a_power_of_two_costs_rotations_and_one_that_is_does_not(self):
        self.assertEqual(metrics.of_qc(build("a = 0..4; return a;")).rotations, 0)
        self.assertGreater(metrics.of_qc(build("a = 0..6; return a;")).rotations, 0)

    def test_stages_are_measured_as_circuits_only_once_the_program_is_qc(self):
        rows = metrics.of_stages(translate(parse("a = 0..4; return a + 1;", "v2")), pipeline.DEFAULT)
        circuits = [circuit for _, _, circuit in rows]
        self.assertIsNone(circuits[0])
        self.assertIsNotNone(circuits[-1])

    def test_a_circuit_fits_a_system_with_enough_qubits_and_otherwise_says_by_how_much_and_what_it_is_made_of(self):
        qc = build("a = 0..4; b = 0..3; return a * b;")
        needed = metrics.fit(qc, System(qubits=1000)).needed
        self.assertEqual(needed, metrics.of_qc(qc).qubits)  # nothing is released early yet, so the peak is the circuit's qubits
        self.assertTrue(metrics.fit(qc, System(qubits=needed)).fits)
        short = metrics.fit(qc, System(qubits=needed - 3))
        self.assertEqual((short.fits, short.over, short.needed), (False, 3, needed))
        self.assertEqual(short.registers, sorted(short.registers, reverse=True))
        self.assertEqual(sum(short.registers), needed)  # every qubit belongs to a register, and the registers are what was allocated

    def test_the_tool_refuses_a_program_that_needs_more_qubits_than_the_system_describes(self):
        source = translate(parse("a = 0..4; b = 0..3; return a * b;", "v2"))
        needed = metrics.fit(pipeline.build(source, *pipeline.DEFAULT), System(qubits=1000)).needed
        pipeline.build(source, *pipeline.DEFAULT, system=System(qubits=needed))  # exactly enough: built
        with self.assertRaises(subprocess.CalledProcessError) as raised:
            pipeline.build(source, *pipeline.DEFAULT, system=System(qubits=needed - 1))
        self.assertIn(f"needs {needed} qubits at once but the system has {needed - 1}", raised.exception.stderr)

    def test_a_run_is_described_by_the_chance_of_each_outcome(self):
        run = metrics.of_run({0: 300, 3: 100})
        self.assertEqual((run.shots, run.outcomes, run.law), (400, 2, {0: 0.75, 3: 0.25}))


if __name__ == "__main__":
    unittest.main()

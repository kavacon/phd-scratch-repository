import unittest
from unittest import mock

from allocation_poc import demo, drawing
from allocation_poc.demo import run_pipeline
from allocation_poc.programs import PROGRAMS
from parser.parser import parse
from passes.allocate import allocate
from passes.isa import lower_to_isa
from passes.qasm import to_openqasm
from utils import qiskit_tools
from utils.text import MAX_TEXT_LINES

needs_qiskit = unittest.skipUnless(qiskit_tools.qiskit_available(), "Qiskit is not installed")
needs_aer = unittest.skipUnless(qiskit_tools.qiskit_available() and qiskit_tools.aer_available(), "Qiskit Aer is not installed")

SMALL = [
    "a = 3; b = a + 2; return b;",
    "a = 2; b = 3; c = a * b; return c;",
    "a = 2; b = 5; c = a < b; return c;",
    "a = 4; b = 4; c = a == b; return c;",
    "a = 5; y = 0; if a < 3 { y = a + 1; } else { y = a - 1; } return y;",
]
ALL = [p.id for p in demo.PASSES]
THROUGH_QASM = ALL[:ALL.index("qasm") + 1]


@needs_qiskit
class QiskitStagesTestCase(unittest.TestCase):
    def circuit(self, source, width=3):
        return to_openqasm(allocate(lower_to_isa(parse(source)), width, 1000))

    def test_the_generated_text_loads_into_qiskit_with_the_same_wires_and_gate_count(self):
        for source in SMALL + [p["source"] for p in PROGRAMS]:
            circuit = self.circuit(source, 6)
            loaded = qiskit_tools.read(circuit.text)
            self.assertEqual(circuit.qubits, loaded.num_qubits)
            self.assertEqual(len(circuit.gates), qiskit_tools.measure(loaded)["gates"], source[:40])

    def test_the_optimised_circuit_uses_only_u_and_cx(self):
        optimised = qiskit_tools.optimise(qiskit_tools.read(self.circuit(SMALL[1]).text))
        self.assertLessEqual(set(qiskit_tools.measure(optimised)["ops"]), {"u", "cx", "measure"})

    def test_measurements_are_counted_apart_from_gates_and_survive_optimisation(self):
        circuit = self.circuit("a = 3; b = a + 2; return b;")
        loaded = qiskit_tools.read(circuit.text)
        before, after = qiskit_tools.measure(loaded), qiskit_tools.measure(qiskit_tools.optimise(loaded))
        self.assertEqual((len(circuit.measured), len(circuit.measured)), (before["measurements"], after["measurements"]))
        self.assertEqual(len(circuit.gates), before["gates"])  # the measurements are not gates
        self.assertEqual(len(circuit.measured), before["ops"]["measure"])

    def test_measure_counts_gates_depth_and_the_wires_that_are_touched(self):
        loaded = qiskit_tools.read("OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[5] q;\nx q[0];\ncx q[0], q[1];\nccx q[0], q[1], q[2];\n")
        measured = qiskit_tools.measure(loaded)
        self.assertEqual((3, 2, 3, 5, 3), (measured["gates"], measured["two_qubit"], measured["depth"], measured["wires"], measured["touched"]))
        self.assertEqual({"x": 1, "cx": 1, "ccx": 1}, measured["ops"])


@needs_qiskit
class CircuitStagesInTheDemoTestCase(unittest.TestCase):
    def stages(self, source, passes=ALL, width=6, qubits=200):
        return {s["id"]: s for s in run_pipeline(source, passes, {"register_width": width, "qubits": qubits})["stages"]}

    def test_each_circuit_stage_reports_gates_depth_and_wires(self):
        stages = self.stages(PROGRAMS[0]["source"])
        for name in ("qasm", "optimise"):
            metrics = stages[name]["circuit"]
            self.assertTrue(stages[name]["single"], name)
            for key in ("gates", "two_qubit", "depth", "wires", "touched", "ops"):
                self.assertIn(key, metrics, name)
            self.assertLessEqual(metrics["touched"], metrics["wires"])
            self.assertIsNone(stages[name]["run"]["peak"], name)  # there are no registers at this level
        self.assertEqual(stages["qasm"]["circuit"]["wires"], stages["optimise"]["circuit"]["wires"])

    def test_circuit_stages_are_not_run_until_asked(self):
        stages = self.stages(PROGRAMS[0]["source"])
        for name in ("qasm", "optimise"):
            self.assertTrue(stages[name]["run"]["notrun"], name)

    @needs_aer
    def test_running_a_circuit_stage_on_request_reports_the_measurement_against_the_expected_value(self):
        options = {"register_width": 6, "qubits": 200}
        outcome = demo.simulate_stage(PROGRAMS[0]["source"], ALL, options, "optimise", 20)
        self.assertEqual(([[13, 20]], 13, True), (outcome["counts"], outcome["expected"], outcome["matches"]))
        self.assertIn("not a circuit stage", demo.simulate_stage(PROGRAMS[0]["source"], ALL, options, "rename")["error"])

    def test_only_the_start_of_a_long_circuit_is_shown(self):
        stage = self.stages(PROGRAMS[0]["source"])["optimise"]  # thousands of gates once rewritten into u and cx
        self.assertLessEqual(len(stage["text"].splitlines()), MAX_TEXT_LINES + 1)
        self.assertIn("more lines", stage["text"].splitlines()[-1])

    def test_a_division_program_stops_at_the_circuit_with_an_explanation(self):
        stage = run_pipeline("a = 8; b = a / 2; return b;", ALL, {"register_width": 6, "qubits": 100})["stages"][-1]
        self.assertEqual("qasm", stage["id"])
        self.assertIn("div is not supported", stage["error"])

    def test_the_rewrite_stages_say_so_when_qiskit_is_missing(self):
        with mock.patch.object(qiskit_tools, "qiskit_available", return_value=False):
            stages = self.stages(PROGRAMS[0]["source"])
        self.assertIsNone(stages["qasm"]["circuit"])
        self.assertIn("Qiskit", stages["optimise"]["error"])


class ServerThreadingTestCase(unittest.TestCase):
    def test_the_demo_server_handles_requests_on_one_thread(self):
        # Qiskit's native allocator crashes the process when a new thread uses it after an earlier thread has finished,
        # which is what a thread per request does, so every request must run on the main thread.
        from socketserver import ThreadingMixIn
        server = demo.make_server(0)
        try:
            self.assertNotIsInstance(server, ThreadingMixIn)
        finally:
            server.server_close()


@needs_qiskit
class DiagramTestCase(unittest.TestCase):
    def stages(self, source, width=3, qubits=200):
        return {s["id"]: s for s in run_pipeline(source, ALL, {"register_width": width, "qubits": qubits})["stages"]}

    def test_the_diagram_labels_each_wire_by_slot_and_bit_with_the_scratch_last(self):
        picture = self.stages("a = 3; b = a + 2; return b;")["qasm"]["diagram"]
        self.assertEqual(["R0[0]", "R0[1]", "R0[2]", "R1[0]", "R1[1]", "R1[2]", "scratch[0]", "scratch[1]", "scratch[2]"], picture["wires"])

    def test_gates_in_the_same_layer_never_share_a_wire_or_cross_each_other(self):
        for stage in ("qasm", "optimise"):
            picture = self.stages(PROGRAMS[0]["source"], 6)[stage]["diagram"]
            by_layer = {}
            for gate in picture["gates"]:
                touched = [gate["target"]] + [q for q, _ in gate["controls"]]
                by_layer.setdefault(gate["layer"], []).append(range(min(touched), max(touched) + 1))
            for spans in by_layer.values():
                covered = [q for span in spans for q in span]
                self.assertEqual(len(covered), len(set(covered)), stage)

    def test_the_generated_circuit_is_drawn_from_its_own_gates_and_the_optimised_one_from_qiskits(self):
        stages = self.stages("a = 3; b = a + 2; return b;")
        generated, optimised = stages["qasm"]["diagram"], stages["optimise"]["diagram"]
        for name, picture in (("qasm", generated), ("optimise", optimised)):  # the diagram holds the measurements too
            metrics = stages[name]["circuit"]
            self.assertEqual(metrics["gates"] + metrics["measurements"], picture["total"], name)
        self.assertEqual({"x", "meter"}, {g["kind"] for g in generated["gates"]})
        self.assertTrue({g["label"] for g in optimised["gates"]} <= {"cx", "U", "M"})

    def test_the_measurements_are_drawn_as_meters_on_the_returned_wires(self):
        for stage in ("qasm", "optimise"):
            picture = self.stages("a = 3; b = a + 2; return b;")[stage]["diagram"]
            meters = [g for g in picture["gates"] if g["kind"] == "meter"]
            self.assertEqual(3, len(meters), stage)
            self.assertEqual({3, 4, 5}, {g["target"] for g in meters}, stage)

    def test_a_long_circuit_is_cut_at_a_fixed_number_of_layers(self):
        picture = self.stages(PROGRAMS[0]["source"], 6)["optimise"]["diagram"]
        self.assertLessEqual(picture["layers"], drawing.MAX_LAYERS)
        self.assertLess(picture["shown"], picture["total"])


if __name__ == '__main__':
    unittest.main()

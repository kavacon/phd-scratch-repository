import unittest

from compiler.passes.qasm import openqasm_text

from checking import qiskit_tools
from compiler.passes.optimise import optimise
from compiler.system import System
from experiments.allocation_poc.experiment import compile_circuit
from experiments.allocation_poc.programs import PROGRAMS
from demo import drawing
from demo.text import MAX_TEXT_LINES, shorten

needs_qiskit = unittest.skipUnless(qiskit_tools.qiskit_available(), "Qiskit is not installed")


class ShortenTestCase(unittest.TestCase):
    def test_only_the_start_of_a_long_text_is_kept_with_a_note_of_the_rest(self):
        text = "\n".join(str(i) for i in range(MAX_TEXT_LINES + 50))
        shown = shorten(text).splitlines()
        self.assertEqual(MAX_TEXT_LINES + 1, len(shown))
        self.assertIn("50 more lines", shown[-1])
        self.assertEqual("a\nb", shorten("a\nb"))


@needs_qiskit
class DiagramTestCase(unittest.TestCase):
    def drawn(self, source, width=3):
        """The diagram of the generated circuit and of Qiskit's optimised version of it, with their counts."""
        circuit = compile_circuit(source, System(width, 200))
        optimised = optimise(circuit)
        return (drawing.diagram(circuit), qiskit_tools.measure(circuit)), (drawing.diagram(optimised), qiskit_tools.measure(optimised))

    def test_the_diagram_labels_each_wire_by_slot_and_bit_with_the_scratch_last(self):
        (picture, _), _ = self.drawn("a = 3; b = a + 2; return b;")
        self.assertEqual(["R0[0]", "R0[1]", "R0[2]", "R1[0]", "R1[1]", "R1[2]", "scratch[0]", "scratch[1]", "scratch[2]"], picture["wires"])

    def test_gates_in_the_same_layer_never_share_a_wire_or_cross_each_other(self):
        for picture, _ in self.drawn(PROGRAMS[0]["source"], 6):
            by_layer = {}
            for gate in picture["gates"]:
                touched = [gate["target"]] + [q for q, _ in gate["controls"]]
                by_layer.setdefault(gate["layer"], []).append(range(min(touched), max(touched) + 1))
            for spans in by_layer.values():
                covered = [q for span in spans for q in span]
                self.assertEqual(len(covered), len(set(covered)))

    def test_the_generated_circuit_is_drawn_with_x_gates_and_the_optimised_one_with_u_and_cx(self):
        (generated, generated_counts), (optimised, optimised_counts) = self.drawn("a = 3; b = a + 2; return b;")
        for picture, counts in ((generated, generated_counts), (optimised, optimised_counts)):  # the diagram holds the measurements too
            self.assertEqual(counts["gates"] + counts["measurements"], picture["total"])
        self.assertEqual({"x", "meter"}, {g["kind"] for g in generated["gates"]})
        self.assertTrue({g["label"] for g in optimised["gates"]} <= {"cx", "U", "M"})

    def test_the_measurements_are_drawn_as_meters_on_the_returned_wires(self):
        for picture, _ in self.drawn("a = 3; b = a + 2; return b;"):
            meters = [g for g in picture["gates"] if g["kind"] == "meter"]
            self.assertEqual(3, len(meters))
            self.assertEqual({3, 4, 5}, {g["target"] for g in meters})

    def test_a_long_circuit_is_cut_at_a_fixed_number_of_layers(self):
        _, (picture, _) = self.drawn(PROGRAMS[0]["source"], 6)
        self.assertLessEqual(picture["layers"], drawing.MAX_LAYERS)
        self.assertLess(picture["shown"], picture["total"])


if __name__ == '__main__':
    unittest.main()

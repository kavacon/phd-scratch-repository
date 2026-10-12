import importlib.util
import unittest

from allocation_poc.programs import PROGRAMS
from utils import qiskit_tools

HAVE_QRISP = importlib.util.find_spec("qrisp") is not None and qiskit_tools.qiskit_available()
needs_qrisp = unittest.skipUnless(HAVE_QRISP, "Qrisp and Qiskit are not installed")


class QrispProgramsTestCase(unittest.TestCase):
    def test_every_demo_program_has_a_qrisp_version_with_a_main(self):
        from allocation_poc import qrisp_programs
        self.assertEqual({p["id"] for p in PROGRAMS}, set(qrisp_programs.SOURCES))
        for program_id, source in qrisp_programs.SOURCES.items():
            self.assertIn("def main():", source, program_id)
            self.assertIn("@auto_uncompute", source, program_id)
            self.assertNotIn("cuccaro", source, program_id)  # built-in operators only
            self.assertIn("from qrisp import QuantumFloat, auto_uncompute", source, program_id)


@needs_qrisp
class QrispRunTestCase(unittest.TestCase):
    def test_a_qrisp_program_compiles_to_a_circuit_with_its_returned_register_measured(self):
        from allocation_poc import qrisp_programs
        built = qrisp_programs.run("chain", 4)
        self.assertEqual(len(built.result), built.circuit.num_clbits)
        self.assertGreaterEqual(len(built.result), 4)  # Qrisp widens the result of arithmetic
        self.assertEqual(built.circuit.num_qubits, len(built.labels))
        self.assertEqual(0, built.exponent)

    def test_the_registers_use_the_width_they_are_given(self):
        from allocation_poc import qrisp_programs
        self.assertGreater(qrisp_programs.run("chain", 6).circuit.num_qubits, qrisp_programs.run("chain", 3).circuit.num_qubits)


@needs_qrisp
class CompareTestCase(unittest.TestCase):
    def test_a_comparison_has_both_sources_and_counts_text_and_a_drawing_for_all_four_circuits(self):
        from allocation_poc import compare
        result = compare.compare("expression", 4)
        self.assertNotIn("error", result)
        self.assertIn("function f", result["sources"]["ours"])
        self.assertIn("@auto_uncompute", result["sources"]["qrisp"])
        self.assertIn("WIDTH = 4", result["sources"]["qrisp"])
        self.assertNotIn("blocks", result)  # the Qrisp programs use Qrisp's own operators, nothing of ours
        for side in ("ours", "qrisp"):
            for variant in ("built", "optimised"):
                entry = result["circuits"][side][variant]
                self.assertGreater(entry["metrics"]["gates"], 0, (side, variant))
                self.assertIn("OPENQASM", entry["text"])
                self.assertTrue(entry["diagram"]["gates"], (side, variant))
        self.assertLessEqual(set(result["circuits"]["ours"]["optimised"]["metrics"]["ops"]), {"u", "cx", "measure"})

    def test_an_unknown_program_is_reported_not_raised(self):
        from allocation_poc import compare
        self.assertIn("message", compare.compare("nothing", 4)["error"])

    def test_the_width_is_kept_in_range(self):
        from allocation_poc import compare
        self.assertEqual((16, 1, 8), (compare.clean_width(99), compare.clean_width(0), compare.clean_width("x")))

    def test_the_server_handles_requests_on_one_thread(self):
        from socketserver import ThreadingMixIn

        from allocation_poc import compare
        server = compare.make_server(0)
        try:
            self.assertNotIsInstance(server, ThreadingMixIn)
        finally:
            server.server_close()


if __name__ == '__main__':
    unittest.main()

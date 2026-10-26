import unittest

from mqt.core.mlir import QCOProgram, QCProgram

OPENQASM = """OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[3] c;
x q[0];
cx q[0], q[1];
ccx q[0], q[1], q[2];
c = measure q;
"""

# A gate carrying an attribute of our own, to see whether annotations can travel through the conversions.
ANNOTATED_QC = """module {
  func.func @main() -> i1 attributes {mqt.entry_point} {
    %alloc = memref.alloc() {mqt.register_name = "q"} : memref<2x!qc.qubit>
    %c0 = arith.constant 0 : index
    %q = memref.load %alloc[%c0] : memref<2x!qc.qubit>
    qc.x %q {xp.role = "compute"} : !qc.qubit
    %b = qc.measure %q : !qc.qubit -> i1
    memref.dealloc %alloc : memref<2x!qc.qubit>
    return %b : i1
  }
}"""

# No quantum operations at all: widened constants, as a front-end would write them.
CLASSICAL = """module {
  func.func @main() -> i32 attributes {mqt.entry_point} {
    %a = arith.constant 5 : i16
    %b = arith.constant 3 : i16
    %x = arith.extsi %a : i16 to i32
    %y = arith.extsi %b : i16 to i32
    %s = arith.addi %x, %y : i32
    return %s : i32
  }
}"""

SUBROUTINE = """OPENQASM 3.0;
include "stdgates.inc";
gate pair a, b { cx a, b; x b; }
qubit[2] q;
bit[2] c;
pair q[0], q[1];
inv @ pair q[0], q[1];
c = measure q;
"""


class MqtWheel(unittest.TestCase):
    """Behaviour of the pinned wheel (mqt.core 4.1.1) that the pipeline relies on; a failure means it changed."""

    def test_printed_qc_text_parses_back_identically(self):
        qc = QCProgram.from_openqasm_str(OPENQASM)
        self.assertEqual(QCProgram.from_mlir_str(qc.ir).ir, qc.ir)

    def test_a_gate_attribute_is_lost_in_the_conversion_to_qco(self):
        qc = QCProgram.from_mlir_str(ANNOTATED_QC)
        qco = qc.to_qco(copy=True)
        self.assertEqual(("xp.role" in qc.ir, "xp.role" in qco.ir, "xp.role" in qco.to_qc(copy=True).ir),
                         (True, False, False))

    def test_upstream_canonicalize_folds_widened_constants(self):
        program = QCOProgram.from_mlir_str(CLASSICAL)
        program.run_pass_pipeline("canonicalize")
        self.assertIn("arith.constant 8 : i32", program.ir)

    def test_a_subroutine_with_its_inverse_survives_conversion_and_export(self):
        qc = QCProgram.from_openqasm_str(SUBROUTINE)
        self.assertEqual(qc.to_qco(copy=True).gate_counts(), {"pair": 1, "inv(pair)": 1, "measure": 2})
        self.assertIn("inv @ pair", qc.to_openqasm3().source)


if __name__ == "__main__":
    unittest.main()

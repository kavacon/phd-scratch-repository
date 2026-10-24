"""
The part of MQT's QC dialect (and its `cbit` classical-bit dialect) that the lowering writes, defined in xDSL so the
lowered program is ordinary IR that our passes can rewrite. Printed in MLIR's generic syntax, MQT's compiler reads it
and prints it back in its own format.

The names, operands and attributes follow what MQT accepts (checked against the pinned wheel by the tests); they are
not a full definition of those dialects.
"""
from xdsl.dialects.builtin import IndexType, IntAttr, StringAttr, SymbolRefAttr, i1
from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.printer import Printer
from xdsl.traits import IsTerminator


@irdl_attr_definition
class QubitType(ParametrizedAttribute, TypeAttribute):
    name = "qc.qubit"


@irdl_attr_definition
class BitRegisterType(ParametrizedAttribute, TypeAttribute):
    """`!cbit.reg<n>`: n classical bits."""

    name = "cbit.reg"
    width: IntAttr

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string(f"<{self.width.data}>")


@irdl_attr_definition
class BitInit(ParametrizedAttribute):
    """`#cbit.init<undefined>`: how a bit register starts."""

    name = "cbit.init"
    kind: StringAttr

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string(f"<{self.kind.data}>")


@irdl_op_definition
class XOp(IRDLOperation):
    name = "qc.x"
    target = operand_def(QubitType)


@irdl_op_definition
class MeasureOp(IRDLOperation):
    name = "qc.measure"
    qubit = operand_def(QubitType)
    result = result_def(i1)


@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "qc.yield"
    traits = traits_def(IsTerminator())


@irdl_op_definition
class CtrlOp(IRDLOperation):
    """The body acts on the targets (the block arguments) and is applied when every control qubit is set."""

    name = "qc.ctrl"
    controls = var_operand_def(QubitType)
    targets = var_operand_def(QubitType)
    body = region_def("single_block")

    irdl_options = (AttrSizedOperandSegments(as_property=True),)


@irdl_op_definition
class CallOp(IRDLOperation):
    name = "qc.call"
    callee = prop_def(SymbolRefAttr)
    qubits = var_operand_def(QubitType)


@irdl_op_definition
class InvOp(IRDLOperation):
    """The inverse of its body, which acts on the block arguments."""

    name = "qc.inv"
    qubits = var_operand_def(QubitType)
    body = region_def("single_block")


@irdl_op_definition
class BitsAllocOp(IRDLOperation):
    name = "cbit.alloc"
    initialization = prop_def(BitInit)
    register = result_def(BitRegisterType)


@irdl_op_definition
class BitsStoreOp(IRDLOperation):
    name = "cbit.store"
    value = operand_def(i1)
    register = operand_def(BitRegisterType)
    index = operand_def(IndexType)


QC = Dialect("qc", [XOp, MeasureOp, YieldOp, CtrlOp, CallOp, InvOp], [QubitType])
CBIT = Dialect("cbit", [BitsAllocOp, BitsStoreOp], [BitRegisterType, BitInit])

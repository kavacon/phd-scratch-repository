"""
Arithmetic on registers, as QC subroutines. An adder is defined once per width and used as a call; subtraction is the
inverse of that call. How a call is expanded, and which adder family sits behind the name, is decided later.
"""
from typing import List, Sequence

from xdsl.dialects import func
from xdsl.dialects.builtin import FunctionType, StringAttr, SymbolRefAttr, UnitAttr
from xdsl.ir import Block, Operation, Region, SSAValue

from experiment_existing_ir.dialects import qc


def cx(control: SSAValue, target: SSAValue) -> Operation:
    return ccx([control], target)


def ccx(controls: Sequence[SSAValue], target: SSAValue) -> Operation:
    """X on the target when every control is set. The body acts on the target, which it receives as an argument."""
    body = Block(arg_types=[qc.QubitType()])
    body.add_ops([qc.XOp(operands=[body.args[0]]), qc.YieldOp()])
    return qc.CtrlOp(operands=[list(controls), [target]], regions=[Region(body)])


def call(name: str, qubits: Sequence[SSAValue]) -> Operation:
    return qc.CallOp(operands=[list(qubits)], properties={"callee": SymbolRefAttr(name)})


def call_inverse(name: str, qubits: Sequence[SSAValue]) -> Operation:
    body = Block(arg_types=[qc.QubitType() for _ in qubits])
    body.add_ops([call(name, body.args), qc.YieldOp()])
    return qc.InvOp(operands=[list(qubits)], regions=[Region(body)])


def adder_name(width: int) -> str:
    return f"add{width}"


def adder(width: int) -> func.FuncOp:
    """
    The ripple-carry adder of Cuccaro et al. on parameters a = 0 .. width-1, b = width .. 2*width-1 and carry = 2*width:
    b becomes (a + b) modulo 2 to the width, and a and the carry (|0> before and after) are unchanged. Two's complement
    needs nothing more, since addition is the same bit for bit whatever the sign.
    """
    block = Block(arg_types=[qc.QubitType() for _ in range(2 * width + 1)])
    a, b, carry = list(block.args[:width]), list(block.args[width:2 * width]), block.args[2 * width]
    carries = [carry] + a[:-1]
    ops: List[Operation] = []
    for i in range(width):  # majority
        ops += [cx(a[i], b[i]), cx(a[i], carries[i]), ccx([carries[i], b[i]], a[i])]
    for i in reversed(range(width)):  # unmajority and add
        ops += [ccx([carries[i], b[i]], a[i]), cx(a[i], carries[i]), cx(carries[i], b[i])]
    block.add_ops(ops + [func.ReturnOp()])
    sub = func.FuncOp(adder_name(width), FunctionType.from_lists([qc.QubitType()] * len(block.args), []),
                      Region(block), visibility=StringAttr("private"))
    sub.attributes["mqt.unitary"] = UnitAttr()
    return sub

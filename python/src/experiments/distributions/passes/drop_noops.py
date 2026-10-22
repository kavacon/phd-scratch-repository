"""
Drop operations that do nothing, and the regions they leave empty: loading zero into a register that is zero, undoing a
load of zero, adding or subtracting zero.

An operation writes into a register that is zero, so loading zero changes nothing, and an inverse of a load of zero expects
zero and leaves it. They arise when a constant is zero, for example the value a variable had before a conditional assigned
it.
"""
from typing import List, Optional

from compiler.compiler import ISA, compile_pass
from compiler.parser.instructions import Const, Ctrl, EndCtrl, ICtrl, Instruction, IOp, IsaProgram, Op, Update
from compiler.system import System


def _nothing(instruction: Instruction) -> bool:
    if isinstance(instruction, (Op, IOp)) and instruction.opcode == "copy" and instruction.sources == (Const(0),):
        return True
    if isinstance(instruction, (Op, IOp)) and instruction.opcode == "copy" and instruction.sources == (Const(False),):
        return True
    return isinstance(instruction, Update) and instruction.source == Const(0)


def _drop(body: List[Instruction]) -> List[Instruction]:
    kept = [i for i in body if not _nothing(i)]
    out: List[Instruction] = []
    for instruction in kept:
        if isinstance(instruction, EndCtrl) and out and isinstance(out[-1], (Ctrl, ICtrl)):
            out.pop()  # an empty region
        else:
            out.append(instruction)
    return out


@compile_pass(
    ISA, ISA,
    title="Drop no-operations",
    summary="Removes loads of zero, additions of zero and the regions they leave empty.",
)
def drop_noops(program: IsaProgram, system: Optional[System] = None) -> IsaProgram:
    body = list(program.body)
    while True:
        shorter = _drop(body)
        if shorter == body:
            return IsaProgram(tuple(body), program.result)
        body = shorter

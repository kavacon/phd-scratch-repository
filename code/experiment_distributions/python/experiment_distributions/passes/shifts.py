"""
Multiplying by a power of two as a shift: `MUL c, b, 2` becomes `SHL c, b, 1`, and its inverse `IMUL` becomes `ISHL`.

A left shift copies each bit of the source one place up and drops the top one, so it is a few CNOTs with no carries.
Its partial inverse is the right shift `SHR`, which reads the destination back as an unsigned number: it undoes the
shift when the dropped bit was zero, and the instruction machine refuses the release when it was not.
"""
from dataclasses import replace
from typing import Optional

from compiler.compiler import ISA, compile_pass
from compiler.parser.instructions import Const, Instruction, IOp, IsaProgram, Op
from compiler.system import System


def _shift(instruction: Instruction) -> Instruction:
    if not (isinstance(instruction, (Op, IOp)) and instruction.opcode == "mul"):
        return instruction
    for variable, constant in (instruction.sources, instruction.sources[::-1]):
        if isinstance(constant, Const) and isinstance(constant.value, int) and not isinstance(constant.value, bool):
            if constant.value >= 2 and constant.value & (constant.value - 1) == 0 and not isinstance(variable, Const):
                return replace(instruction, opcode="shl", sources=(variable, Const(constant.value.bit_length() - 1)))
    return instruction


@compile_pass(
    ISA, ISA,
    title="Multiply by shifting",
    summary="Rewrites a multiplication by a power of two as a left shift.",
)
def shift_multiplications(program: IsaProgram, system: Optional[System] = None) -> IsaProgram:
    return replace(program, body=tuple(_shift(i) for i in program.body))

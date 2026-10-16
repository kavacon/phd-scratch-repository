"""
One instruction on its own, as a tiny allocated program: the operands and the destination are registers, the scratch
register follows them, and the program returns the destination. Each lowering turns it into a circuit that holds only
that instruction's gates, which is how to look at what an operation costs in each encoding.

A variant says the opcode and which operands are registers and which are the constant.
"""
from dataclasses import dataclass
from typing import List, Tuple

from compiler.parser.instructions import Alloc, Const, IsaProgram, Op, Operand, Reg, Scratch, Update


@dataclass(frozen=True)
class Variant:
    id: str
    name: str
    description: str
    opcode: str
    operands: str  # one letter per source: "r" for a register, "c" for the constant
    update: bool = False  # in place: dst += source or dst -= source

    def program(self, constant: int, returns: bool = False) -> IsaProgram:
        """
        The instruction on fresh registers, named in the order they appear, then the scratch register. With no result
        it is just the instruction, and a lowering leaves the destination as the instruction does; `returns` makes the
        program return the destination, so a lowering also reads it out.
        """
        registers: List[str] = []

        def operand(kind: str) -> Operand:
            if kind == "c":
                return Const(constant)
            registers.append(f"R{len(registers)}")
            return Reg(registers[-1])

        if self.update:
            dst = Reg("R0")
            registers.append("R0")
            source = operand(self.operands)
            instruction = Update(self.opcode, dst.name, source)
        else:
            sources = tuple(operand(kind) for kind in self.operands)
            dst = Reg(f"R{len(registers)}")
            registers.append(dst.name)
            instruction = Op(self.opcode, dst.name, sources)
        body: Tuple = (Scratch(f"R{len(registers)}"), *(Alloc(r) for r in registers), instruction)
        return IsaProgram(body, dst if returns else None)


def _v(opcode, operands, name, description, update=False) -> Variant:
    return Variant(f"{'update-' if update else ''}{opcode}-{operands}", name, description, opcode, operands, update)


VARIANTS = (
    _v("add", "rr", "ADD reg, reg", "dst = a + b with both operands in registers."),
    _v("add", "rc", "ADD reg, const", "dst = a + the constant."),
    _v("sub", "rr", "SUB reg, reg", "dst = a - b with both operands in registers."),
    _v("sub", "rc", "SUB reg, const", "dst = a - the constant."),
    _v("sub", "cr", "SUB const, reg", "dst = the constant - b."),
    _v("mul", "rr", "MUL reg, reg", "dst = a * b with both operands in registers."),
    _v("mul", "rc", "MUL reg, const", "dst = a * the constant."),
    _v("add", "r", "ADD= reg", "In place: dst += a register.", update=True),
    _v("add", "c", "ADD= const", "In place: dst += the constant.", update=True),
    _v("sub", "r", "SUB= reg", "In place: dst -= a register.", update=True),
    _v("sub", "c", "SUB= const", "In place: dst -= the constant.", update=True),
    _v("lt", "rr", "LT reg, reg", "dst = a < b, signed."),
    _v("lt", "rc", "LT reg, const", "dst = a < the constant, signed."),
    _v("lt", "cr", "LT const, reg", "dst = the constant < b, signed."),
    _v("gt", "rr", "GT reg, reg", "dst = a > b, signed."),
    _v("gt", "rc", "GT reg, const", "dst = a > the constant, signed."),
    _v("eq", "rr", "EQ reg, reg", "dst = a == b."),
    _v("eq", "rc", "EQ reg, const", "dst = a == the constant."),
    _v("ne", "rr", "NE reg, reg", "dst = a != b."),
    _v("ne", "rc", "NE reg, const", "dst = a != the constant."),
    _v("and", "rr", "AND", "dst = a and b, on bit 0 of each register."),
    _v("or", "rr", "OR", "dst = a or b, on bit 0 of each register."),
    _v("not", "r", "NOT", "dst = not a, on bit 0."),
    _v("copy", "r", "COPY reg", "dst = a copy of a register."),
    _v("copy", "c", "COPY const", "dst = the constant, loaded with X gates."),
)

"""
Physical allocation: rewrite an instruction program so every logical register lives in a numbered slot.

The machine is described by two compile parameters, the number of qubits in one register (`register_width`) and
the total number of qubits to compute with (`qubits`). One register's worth of qubits is reserved as scratch, so it
has `qubits // register_width - 1` slots, and each slot holds one register. Logical registers are never reused by
the earlier passes, so a slot is the unit of reuse here.

Scratch. Operations such as ADD and MUL may need ancilla qubits while they run. One shared scratch register, as wide
as a register, is reserved for them, declared with `SCRATCH` after the slots, and is never handed out as a slot, so no
operation allocates inside a controlled region. This rests on assumptions about whatever circuit implements an opcode, which any such circuit must keep:
it needs at most `register_width` scratch qubits, and it returns them to zero before it finishes (releasing the
scratch is part of the circuit's contract), so the next operation finds it clean. One scratch register means two
operations cannot use scratch at the same time; that does not matter for a sequential list, but a later pass that
runs operations in parallel would have to treat it as a single shared resource. The reservation is made whether or
not the program has any such operation, which can waste qubits.

One pass over the instruction list, with a pool of free slots:
- `ALLOC r` takes the lowest free slot, or opens a new one if there is none.
- `FREE r` gives its slot back. The register is zero by then, so the next owner needs no reset.
- Every other instruction has its registers renamed to their slots (`R0`, `R1`, ...).
For a fixed instruction order the number of slots opened equals the most registers held at once, which is the
fewest any packing can use, so this is optimal for the order it is given. A program that needs more slots than the
machine has raises `WidthExceeded`; lowering the peak takes rewrites that change the order or the program.

Functions were inlined earlier, so there are no calls to treat specially. The result is an ordinary instruction
program on the physical registers, with the scratch register declared by a `SCRATCH` instruction, so it has the same
shape as its input: instruction `k` of the result is instruction `k` of the input, after the declaration. What each slot
held at every step can be read off by walking the two programs together.
"""
from typing import Dict, List, Optional

from compiler.compiler import ISA, compile_pass
from compiler.parser.instructions import (
    Alloc,
    Const,
    Ctrl,
    EndCtrl,
    Free,
    ICtrl,
    Instruction,
    IOp,
    IsaProgram,
    Op,
    Operand,
    Reg,
    Scratch,
    Update,
)
from compiler.system import System


class AllocationError(Exception):
    pass


class WidthExceeded(AllocationError):
    """The program needs more slots than the machine has. `partial` is the packing it would have needed."""

    def __init__(self, needed: int, available: int, partial: IsaProgram):
        super().__init__(f"needs {needed} register slots but only {available} are available")
        self.needed = needed
        self.available = available
        self.partial = partial  # what the compiler shows for the stage that failed


def slot_name(slot: int) -> str:
    return f"R{slot}"


class _Allocator:
    """Walks a program once, giving each logical register the lowest free slot and rewriting every use of it."""

    def __init__(self) -> None:
        self.free: List[int] = []
        self.opened = 0
        self.operands: Dict[str, Operand] = {}

    def operand(self, operand: Operand) -> Operand:
        if isinstance(operand, Const):
            return operand
        if operand.name not in self.operands:
            raise AllocationError(f"{operand.name} is used before it is allocated")
        return self.operands[operand.name]

    def register(self, name: str) -> str:
        mapped = self.operands.get(name)
        if not isinstance(mapped, Reg):
            raise AllocationError(f"{name} is written before it is allocated")
        return mapped.name

    def alloc(self, instruction: Alloc) -> Instruction:
        if self.free:
            slot = min(self.free)
            self.free.remove(slot)
        else:
            slot, self.opened = self.opened, self.opened + 1
        self.operands[instruction.register] = Reg(slot_name(slot))
        return Alloc(slot_name(slot))

    def release(self, instruction: Free) -> Instruction:
        name = self.register(instruction.register)
        self.free.append(int(name[1:]))
        return Free(name)

    def rewrite(self, instruction: Instruction) -> Instruction:
        """The instruction on physical slots."""
        if isinstance(instruction, Alloc):
            return self.alloc(instruction)
        if isinstance(instruction, Free):
            return self.release(instruction)
        if isinstance(instruction, (Op, IOp)):
            sources = tuple(self.operand(source) for source in instruction.sources)
            return type(instruction)(instruction.opcode, self.register(instruction.dst), sources)
        if isinstance(instruction, Update):
            return Update(instruction.opcode, self.register(instruction.dst), self.operand(instruction.source))
        if isinstance(instruction, (Ctrl, ICtrl)):
            return type(instruction)(self.operand(instruction.control))
        if isinstance(instruction, EndCtrl):
            return instruction
        raise AllocationError(f"cannot allocate {type(instruction).__name__}")


@compile_pass(
    ISA, ISA,
    title="Assign physical qubits",
    summary=(
        "Maps each logical register to a numbered slot, reusing a slot once its register is freed, and "
        "declares the scratch register."
    ),
)
def allocate(program: IsaProgram, system: Optional[System] = None) -> IsaProgram:
    """
    Rewrite `program` onto numbered slots and declare the scratch register after them (`SCRATCH R<n>`). Raises
    `WidthExceeded`, carrying the packing it would have needed, if the program needs more slots than the system has.
    """
    system = system or System()
    allocator = _Allocator()
    body: List[Instruction] = [allocator.rewrite(instruction) for instruction in program.body]
    result = IsaProgram((Scratch(slot_name(allocator.opened)), *body), None if program.result is None else allocator.operand(program.result))
    if allocator.opened > system.slots:
        raise WidthExceeded(allocator.opened, system.slots, result)
    return result

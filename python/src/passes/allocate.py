"""
Physical allocation: rewrite an instruction program so every logical register lives in a numbered slot.

The machine is described by two compile parameters, the number of qubits in one register (`register_width`) and
the total number of qubits to compute with (`qubits`). One register's worth of qubits is reserved as scratch, so it
has `qubits // register_width - 1` slots, and each slot holds one register. Logical registers are never reused by
the earlier passes, so a slot is the unit of reuse here.

Scratch. Operations such as ADD and MUL may need ancilla qubits while they run. One shared scratch register, as wide
as a register, is reserved for them and is never handed out as a slot, so no operation allocates inside a controlled
region. This rests on assumptions about whatever circuit implements an opcode, which any such circuit must keep:
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

Functions were inlined earlier, so there are no calls to treat specially. `Allocation.trace` records the slots after
every instruction so a viewer can show what is held where.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from parser.instructions import (
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
)


class AllocationError(Exception):
    pass


class WidthExceeded(AllocationError):
    """The program needs more slots than the machine has. `allocation` is the packing it would have needed."""

    def __init__(self, needed: int, available: int, allocation: "Allocation"):
        super().__init__(f"needs {needed} register slots but only {available} are available")
        self.needed = needed
        self.available = available
        self.allocation = allocation


@dataclass(frozen=True)
class Held:
    """A logical register sitting in a slot."""
    slot: int
    register: str


@dataclass(frozen=True)
class Step:
    """One instruction as it ran: the logical and physical forms and what every slot held afterwards."""
    logical: Instruction
    physical: Instruction
    held: Tuple[Held, ...]
    changed: Optional[int]  # the slot this instruction took or gave back, if any


@dataclass(frozen=True)
class Allocation:
    program: IsaProgram
    register_width: int
    qubits: int
    slots: int  # how many register slots the machine has, after the scratch register is set aside
    needed: int  # how many the program opened
    trace: Tuple[Step, ...]

    @property
    def scratch_qubits(self) -> int:
        return self.register_width

    @property
    def qubits_used(self) -> int:
        return self.needed * self.register_width + self.scratch_qubits

    @property
    def fits(self) -> bool:
        return self.needed <= self.slots


def slot_name(slot: int) -> str:
    return f"R{slot}"


class _Allocator:
    """Walks a program once, giving each logical register the lowest free slot and rewriting every use of it."""

    def __init__(self) -> None:
        self.free: List[int] = []
        self.opened = 0
        self.held: Dict[int, Held] = {}
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

    def alloc(self, instruction: Alloc) -> Tuple[Instruction, int]:
        if self.free:
            slot = min(self.free)
            self.free.remove(slot)
        else:
            slot, self.opened = self.opened, self.opened + 1
        self.held[slot] = Held(slot, instruction.register)
        self.operands[instruction.register] = Reg(slot_name(slot))
        return Alloc(slot_name(slot)), slot

    def release(self, instruction: Free) -> Tuple[Instruction, int]:
        name = self.register(instruction.register)
        slot = int(name[1:])
        del self.held[slot]
        self.free.append(slot)
        return Free(name), slot

    def rewrite(self, instruction: Instruction) -> Tuple[Instruction, Optional[int]]:
        """The instruction on physical slots, and the slot it opened or closed, if any."""
        if isinstance(instruction, Alloc):
            return self.alloc(instruction)
        if isinstance(instruction, Free):
            return self.release(instruction)
        if isinstance(instruction, (Op, IOp)):
            sources = tuple(self.operand(source) for source in instruction.sources)
            return type(instruction)(instruction.opcode, self.register(instruction.dst), sources), None
        if isinstance(instruction, (Ctrl, ICtrl)):
            return type(instruction)(self.operand(instruction.control)), None
        if isinstance(instruction, EndCtrl):
            return instruction, None
        raise AllocationError(f"cannot allocate {type(instruction).__name__}")

    def snapshot(self) -> Tuple[Held, ...]:
        return tuple(sorted(self.held.values(), key=lambda held: held.slot))


def allocate(program: IsaProgram, register_width: int, qubits: int) -> Allocation:
    """Rewrite `program` onto numbered slots. Raises `WidthExceeded` if it needs more than the machine has."""
    if register_width < 1:
        raise ValueError("register_width must be at least 1")
    if qubits < 0:
        raise ValueError("qubits cannot be negative")

    allocator = _Allocator()
    body: List[Instruction] = []
    trace: List[Step] = []
    for instruction in program.body:
        physical, changed = allocator.rewrite(instruction)
        body.append(physical)
        trace.append(Step(instruction, physical, allocator.snapshot(), changed))

    slots = max(qubits // register_width - 1, 0)  # one register is kept back for scratch
    result = Allocation(IsaProgram(tuple(body), allocator.operand(program.result)), register_width, qubits, slots, allocator.opened, tuple(trace))
    if not result.fits:
        raise WidthExceeded(result.needed, result.slots, result)
    return result

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
    Alloc, Const, Ctrl, EndCtrl, Free, ICtrl, IOp, Instruction, IsaProgram, Op, Operand, Reg,
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


def allocate(program: IsaProgram, register_width: int, qubits: int) -> Allocation:
    """Rewrite `program` onto numbered slots. Raises `WidthExceeded` if it needs more than the machine has."""
    if register_width < 1:
        raise ValueError("register_width must be at least 1")
    if qubits < 0:
        raise ValueError("qubits cannot be negative")

    free: List[int] = []
    opened = 0
    held: Dict[int, Held] = {}
    operands: Dict[str, Operand] = {}
    trace: List[Step] = []

    def operand(o: Operand) -> Operand:
        if isinstance(o, Const):
            return o
        if o.name not in operands:
            raise AllocationError(f"{o.name} is used before it is allocated")
        return operands[o.name]

    def register(name: str) -> str:
        mapped = operands.get(name)
        if not isinstance(mapped, Reg):
            raise AllocationError(f"{name} is written before it is allocated")
        return mapped.name

    body: List[Instruction] = []
    for instruction in program.body:
        changed: Optional[int] = None
        if isinstance(instruction, Alloc):
            if free:
                changed = min(free)
                free.remove(changed)
            else:
                changed, opened = opened, opened + 1
            held[changed] = Held(changed, instruction.register)
            operands[instruction.register] = Reg(slot_name(changed))
            physical: Instruction = Alloc(slot_name(changed))
        elif isinstance(instruction, Free):
            name = register(instruction.register)
            changed = int(name[1:])
            del held[changed]
            free.append(changed)
            physical = Free(name)
        elif isinstance(instruction, (Op, IOp)):
            physical = type(instruction)(instruction.opcode, register(instruction.dst), tuple(operand(s) for s in instruction.sources))
        elif isinstance(instruction, (Ctrl, ICtrl)):
            physical = type(instruction)(operand(instruction.control))
        elif isinstance(instruction, EndCtrl):
            physical = instruction
        else:
            raise AllocationError(f"cannot allocate {type(instruction).__name__}")
        body.append(physical)
        trace.append(Step(instruction, physical, tuple(sorted(held.values(), key=lambda h: h.slot)), changed))

    slots = max(qubits // register_width - 1, 0)  # one register is kept back for scratch
    result = Allocation(IsaProgram(tuple(body), operand(program.result)), register_width, qubits, slots, opened, tuple(trace))
    if not result.fits:
        raise WidthExceeded(result.needed, result.slots, result)
    return result

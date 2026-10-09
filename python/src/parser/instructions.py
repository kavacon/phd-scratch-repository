"""
The pre-circuit instruction set that a lowered program is written in. Registers are logical (one per version,
no widths yet) and an instruction becomes a circuit fragment directly.

- `Alloc r` takes a clean (zero) register and `Free r` gives back one that is zero. They mark where a register
  lives, which is what a later register allocation pass reads.
- `Op opcode dst, sources` is `dst ^= opcode(sources)`. A register starts at zero, so this writes the result,
  and applying the same instruction again cancels it, so the instruction is its own inverse.
- `Ctrl c` makes every instruction after it controlled on register `c`, and `InvCtrl c` does the same on the
  negation of `c`. `EndCtrl` ends the most recent one that is still open, so regions nest and each extra open one
  adds a control. There are no jumps and no blocks: an `if` is a `Ctrl` region and an `InvCtrl` region, and both
  are always in the circuit. A controlled region never allocates or frees; those are placed before and after it.
- `Call function, dst, args` is `dst ^= function(args)`: it names another instruction set, the way a call jumps to
  a memory location. It is kept as an instruction because it is not yet known how a QPU or hybrid device would run
  a call, so inlining and the like are left to circuit generation.

A program is several instruction sets, one for `main` and one for each function. Each is a flat list of these
instructions, with registers of its own, and ends in the value it returns.

Each instruction has an inverse (`inverse`), which is what reversing a history uses. Instructions carry no
flags: whether one is an undo is a fact about the history, not about the circuit. In-place forms (`x += y`,
inverse `x -= y`) are not defined yet.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union


@dataclass(frozen=True)
class Reg:
    name: str


@dataclass(frozen=True)
class Const:
    value: Union[int, float, bool]


Operand = Union[Reg, Const]

OPCODES = ("copy", "add", "sub", "mul", "div", "lt", "gt", "eq", "ne", "and", "or", "not")


@dataclass(frozen=True)
class Instruction:
    pass


@dataclass(frozen=True)
class Alloc(Instruction):
    register: str


@dataclass(frozen=True)
class Free(Instruction):
    register: str


@dataclass(frozen=True)
class Op(Instruction):
    opcode: str
    dst: str
    sources: Tuple[Operand, ...]


@dataclass(frozen=True)
class Ctrl(Instruction):
    """Start a region controlled on a register."""
    control: Operand


@dataclass(frozen=True)
class InvCtrl(Instruction):
    """Start a region controlled on the negation of a register."""
    control: Operand


@dataclass(frozen=True)
class EndCtrl(Instruction):
    """End the most recently opened region."""


@dataclass(frozen=True)
class Call(Instruction):
    function: str
    dst: Optional[str]
    args: Tuple[Operand, ...]


@dataclass(frozen=True)
class InstructionSet:
    """One flat list of instructions (`main`, or a function) and the value it returns."""
    name: str
    params: Tuple[str, ...]
    body: Tuple[Instruction, ...]
    result: Operand


@dataclass(frozen=True)
class IsaProgram:
    functions: Tuple[InstructionSet, ...]
    main: InstructionSet


def inverse(instruction: Instruction) -> Instruction:
    """The inverse of one instruction. A region marker has none on its own; see `inverse_block`."""
    if isinstance(instruction, Alloc):
        return Free(instruction.register)
    if isinstance(instruction, Free):
        return Alloc(instruction.register)
    if isinstance(instruction, (Op, Call)):
        return instruction  # XOR into a register is its own inverse
    raise TypeError(f"no inverse for {type(instruction).__name__} on its own")


def inverse_block(block: Tuple[Instruction, ...]) -> Tuple[Instruction, ...]:
    """
    Undoes a sequence: the inverse of each instruction, in reverse order. A controlled region keeps its control
    but runs backwards, so where it opened it now closes and where it closed it now opens.
    """
    opened = []  # the opener of each region that is still open
    opener_of = {}  # index of an `EndCtrl` -> its opener
    for k, instruction in enumerate(block):
        if isinstance(instruction, (Ctrl, InvCtrl)):
            opened.append(instruction)
        elif isinstance(instruction, EndCtrl):
            if not opened:
                raise ValueError("EndCtrl with no open region")
            opener_of[k] = opened.pop()
    if opened:
        raise ValueError("a controlled region is never closed")
    result = []
    for k in reversed(range(len(block))):
        instruction = block[k]
        if isinstance(instruction, EndCtrl):
            result.append(opener_of[k])
        elif isinstance(instruction, (Ctrl, InvCtrl)):
            result.append(EndCtrl())
        else:
            result.append(inverse(instruction))
    return tuple(result)

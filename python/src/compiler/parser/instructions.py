"""
The pre-circuit instruction set that a lowered program is written in. Registers are logical (one per version,
no widths yet) and an instruction becomes a circuit fragment directly.

- `Alloc r` takes a clean (zero) register and `Free r` gives back one that is zero. They mark where a register
  lives, which is what a later register allocation pass reads.
- `Op opcode dst, sources` (for example `MUL p, x, y`) needs `dst` to be zero and leaves it holding
  `opcode(sources)`. `IOp` is its explicit inverse (`IMUL p, x, y`): it needs `dst` to equal `opcode(sources)` and
  leaves it zero. The `I` prefix is the same one `ICtrl` uses. Nothing is implied: a release is written as the
  inverse instruction, and a circuit for it is the reversed circuit of the instruction it undoes. Both
  preconditions can be checked, which is also how a manually written `FREE` or release can be checked.
- `Ctrl c` makes every instruction after it controlled on register `c`, and `ICtrl c` does the same on the
  negation of `c`. `EndCtrl` ends the most recent one that is still open, so regions nest and each extra open one
  adds a control. There are no jumps and no blocks: an `if` is a `Ctrl` region and an `ICtrl` region, and both
  are always in the circuit. A controlled region never allocates or frees; those are placed before and after it.
- `Dist r, start, stop` takes a zero register and leaves it in an equal superposition of the integers from `start` up to,
  not including, `stop`. It is the one place a superposition is created: how it is prepared (Hadamards for a power of two,
  something more for other sizes) is the circuit lowering's concern, and which registers end up superposed or entangled
  is read off the instruction list, not stored. `IDist` is its inverse: it needs `r` to hold exactly that state and
  leaves it zero.

A program is one flat list of these instructions and, usually, the value it returns. A program with no `result` is just
a body: a lowering builds its instructions and nothing to read the result out. There is no call instruction: calls were
descoped for this version, and the inline pass has already copied every callee into its call sites, so a program
reaches this level with no functions. A `CALL` to a separate instruction set could return if a target needs
subroutines to stay first-class.

Every instruction has an explicit inverse (`inverse`): `ALLOC`/`FREE`, `OP`/`IOP`, `DIST`/`IDIST`, and a controlled region reversed.
In-place forms (`x += y`, whose inverse `x -= y` is a different circuit) are not defined yet.
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
class Scratch(Instruction):
    """
    Declares which register is the machine's scratch register. Operations may borrow its qubits while they run (and
    must leave it zero); no instruction reads or writes it, and it is never allocated or freed.
    """
    register: str


@dataclass(frozen=True)
class Op(Instruction):
    opcode: str
    dst: str
    sources: Tuple[Operand, ...]


@dataclass(frozen=True)
class Dist(Instruction):
    """`dst` is zero and becomes an equal superposition of the integers from `start` up to, not including, `stop`."""
    dst: str
    start: int
    stop: int


@dataclass(frozen=True)
class IDist(Instruction):
    """The inverse of `Dist` with the same fields: `dst` holds exactly that superposition and is left zero."""
    dst: str
    start: int
    stop: int


@dataclass(frozen=True)
class Update(Instruction):
    """
    `dst += source` or `dst -= source` in place (`opcode` is "add" or "sub"). Unlike `Op` it needs no zero register:
    `dst` holds a value and keeps the result. Its inverse is the other opcode, so a program can undo it exactly.
    """
    opcode: str
    dst: str
    source: Operand


@dataclass(frozen=True)
class IOp(Instruction):
    """The inverse of `Op` with the same fields: `dst` equals `opcode(sources)` and is left zero."""
    opcode: str
    dst: str
    sources: Tuple[Operand, ...]


@dataclass(frozen=True)
class Ctrl(Instruction):
    """Start a region controlled on a register."""
    control: Operand


@dataclass(frozen=True)
class ICtrl(Instruction):
    """Start a region controlled on the negation of a register."""
    control: Operand


@dataclass(frozen=True)
class EndCtrl(Instruction):
    """End the most recently opened region."""


@dataclass(frozen=True)
class IsaProgram:
    body: Tuple[Instruction, ...]
    result: Optional[Operand] = None


def inverse(instruction: Instruction) -> Instruction:
    """The inverse of one instruction. A region marker has none on its own; see `inverse_block`."""
    if isinstance(instruction, Alloc):
        return Free(instruction.register)
    if isinstance(instruction, Free):
        return Alloc(instruction.register)
    if isinstance(instruction, Op):
        return IOp(instruction.opcode, instruction.dst, instruction.sources)
    if isinstance(instruction, IOp):
        return Op(instruction.opcode, instruction.dst, instruction.sources)
    if isinstance(instruction, Dist):
        return IDist(instruction.dst, instruction.start, instruction.stop)
    if isinstance(instruction, IDist):
        return Dist(instruction.dst, instruction.start, instruction.stop)
    if isinstance(instruction, Update):
        return Update({"add": "sub", "sub": "add"}[instruction.opcode], instruction.dst, instruction.source)
    raise TypeError(f"no inverse for {type(instruction).__name__} on its own")


def inverse_block(block: Tuple[Instruction, ...]) -> Tuple[Instruction, ...]:
    """
    Undoes a sequence: the inverse of each instruction, in reverse order. A controlled region keeps its control
    but runs backwards, so where it opened it now closes and where it closed it now opens.
    """
    opened = []  # the opener of each region that is still open
    opener_of = {}  # index of an `EndCtrl` -> its opener
    for k, instruction in enumerate(block):
        if isinstance(instruction, (Ctrl, ICtrl)):
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
        elif isinstance(instruction, (Ctrl, ICtrl)):
            result.append(EndCtrl())
        else:
            result.append(inverse(instruction))
    return tuple(result)

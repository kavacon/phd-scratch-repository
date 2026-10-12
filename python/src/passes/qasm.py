"""
Circuit generation: turn an allocated instruction program into an OpenQASM 3 circuit.

Layout. Slot `i` is the qubits `i*w` to `i*w + w - 1` of one register `q`, least significant bit first, where `w` is
the register width. The shared scratch register follows the slots, so the circuit has `(slots used + 1) * w` qubits.
An integer is unsigned and arithmetic is modulo 2^w. A boolean (a comparison, `and`, `or`, `not`) is bit 0 of its
register with the other bits zero; `CTRL` and the logical operations read only bit 0.

Every gate here is an X with some controls (X, CNOT, Toffoli, multi-controlled X), each control positive or
negative, so a circuit is a list of `Gate`s. Each opcode writes straight into its destination, which starts at
zero, and its inverse instruction is the same gates in reverse order (every gate undoes itself), so the circuit of
`IMUL` is the reversed circuit of `MUL`. A `CTRL` region adds its control to every gate inside it.

The circuits:
- `copy` is one CNOT per bit (X for a constant 1 bit). `not`, `and` and `or` act on bit 0 with X, CNOT and Toffoli.
- `add` copies the first operand into the destination and then adds the second in place with a Cuccaro ripple-carry
  adder (MAJ and UMA gates, one carry qubit taken from scratch, restored afterwards). `sub` copies the first operand
  and runs the adder backwards. A constant is added with a cascade of multi-controlled X gates that increment the
  destination, which needs no qubit for the constant.
- `mul` is shift-and-add: for each bit of one operand, a controlled add of the other operand shifted into the
  destination. A constant operand turns the controls into plain shifted adds. Squaring a register (`x * x`) would
  need a copy of it and is not supported yet.
- `eq` XORs the operands in place, flips the destination if every bit is then zero, and undoes the XOR; `ne` flips
  the destination after it. A constant needs only one multi-controlled X with mixed polarity controls.
- `lt` copies the borrow out of `a - b`, computed as the carry of `(not a) + b` along a MAJ chain that is run
  backwards afterwards; `gt` swaps the operands. A constant needs only a few mixed-polarity multi-controlled X gates.
- `div` is not supported.
Measurement. The program's returned register is measured at the end into a classical register `result`, bit `i` from
qubit `i`; nothing else is measured, because everything else has been returned to zero. A program always returns a
register: flatten names a returned literal, so `return 2;` is a variable holding 2 like any other. Operations on a
constant alone are computed here and written as X gates. The circuits assume the scratch qubits
start at zero and are returned to zero by every operation, which is the contract the scratch register relies on.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

from parser.instructions import (
    Alloc,
    Const,
    Ctrl,
    EndCtrl,
    Free,
    ICtrl,
    IOp,
    Op,
    Operand,
    Reg,
)
from passes.allocate import Allocation


class CircuitError(Exception):
    pass


@dataclass(frozen=True)
class Q:
    """A wire."""
    index: int


@dataclass(frozen=True)
class C:
    """A classical bit known at compile time."""
    value: int


Bit = Union[Q, C]


@dataclass(frozen=True)
class Gate:
    """X on `target`, applied when every control holds (a control is (qubit, polarity))."""
    target: int
    controls: Tuple[Tuple[int, bool], ...] = ()


@dataclass(frozen=True)
class Circuit:
    text: str
    qubits: int
    register_width: int
    gates: Tuple[Gate, ...]
    result: Tuple[Bit, ...]  # the returned value, least significant bit first
    measured: Tuple[int, ...] = ()  # the qubits measured at the end, one per bit of a returned register


def gate_text(gate: Gate) -> str:
    positive = [q for q, polarity in gate.controls if polarity]
    negative = [q for q, polarity in gate.controls if not polarity]
    wires = ", ".join(f"q[{q}]" for q in positive + negative + [gate.target])
    if not negative and len(positive) <= 2:
        return f"{['x', 'cx', 'ccx'][len(positive)]} {wires};"
    parts = []
    for name, group in (("ctrl", positive), ("negctrl", negative)):
        if group:
            parts.append(name if len(group) == 1 else f"{name}({len(group)})")
    return f"{' @ '.join(parts)} @ x {wires};"


def _majority(c: int, b: int, a: int) -> List[Gate]:
    return [Gate(b, ((a, True),)), Gate(c, ((a, True),)), Gate(a, ((c, True), (b, True)))]


def _unmajority(c: int, b: int, a: int) -> List[Gate]:
    return [Gate(a, ((c, True), (b, True))), Gate(c, ((a, True),)), Gate(b, ((c, True),))]


def adder(acc: Sequence[int], add: Sequence[int], carry: int) -> List[Gate]:
    """acc += add (modulo 2^n) with a Cuccaro adder; `add` and `carry` are restored."""
    n = len(acc)
    if n == 0:
        return []
    if n == 1:
        return [Gate(acc[0], ((add[0], True),))]
    carries = [carry] + list(add[:n - 1])
    gates: List[Gate] = []
    for i in range(n - 1):
        gates += _majority(carries[i], acc[i], add[i])
    gates.append(Gate(acc[n - 1], ((add[n - 1], True),)))
    gates.append(Gate(acc[n - 1], ((add[n - 2], True),)))
    for i in reversed(range(n - 1)):
        gates += _unmajority(carries[i], acc[i], add[i])
    return gates


def _same(a: Sequence[Bit], b: Sequence[Bit]) -> bool:
    return all(isinstance(x, Q) for x in a) and list(a) == list(b)


def _constant(bits: Sequence[Bit]) -> Optional[int]:
    if all(isinstance(b, C) for b in bits):
        return sum(b.value << i for i, b in enumerate(bits))
    return None


class _Builder:
    """The gates of one instruction."""

    def __init__(self, width: int, scratch: int) -> None:
        self.w = width
        self.scratch = scratch
        self.gates: List[Gate] = []

    def gate(self, target: int, controls: Sequence[Tuple[Bit, bool]] = ()) -> None:
        kept = []
        for bit, polarity in controls:
            if isinstance(bit, C):
                if bool(bit.value) != polarity:
                    return  # a control that can never hold, so the gate does nothing
            else:
                kept.append((bit.index, polarity))
        self.gates.append(Gate(target, tuple(kept)))

    def load(self, dst: Sequence[int], value: int) -> None:
        for i, q in enumerate(dst):
            if (value >> i) & 1:
                self.gate(q)

    def copy(self, dst: Sequence[int], src: Sequence[Bit]) -> None:
        for q, bit in zip(dst, src, strict=True):
            self.gate(q, [(bit, True)])

    def add_constant(self, acc: Sequence[int], value: int) -> None:
        """acc += value by incrementing from each set bit of value upwards."""
        n = len(acc)
        for j in range(n):
            if (value >> j) & 1:
                for i in reversed(range(j + 1, n)):
                    self.gate(acc[i], [(Q(acc[k]), True) for k in range(j, i)])
                self.gate(acc[j])

    def shifted_add(self, dst: Sequence[int], src: Sequence[int], shift: int, controls: Sequence[Tuple[Bit, bool]] = ()) -> None:
        n = self.w - shift
        for g in adder(dst[shift:], src[:n], self.scratch):
            self.gates.append(Gate(g.target, g.controls + tuple((b.index, p) for b, p in controls)))

    def carry_chain(self, a: Sequence[int], b: Sequence[int]) -> List[Gate]:
        """After these gates the carry out of a + b sits where a's top bit was; they are undone by reversing them."""
        carries = [self.scratch] + list(a[:-1])
        gates: List[Gate] = []
        for i in range(len(a)):
            gates += _majority(carries[i], b[i], a[i])
        return gates

    # -- opcodes ------------------------------------------------------------------------------------------------

    def op(self, opcode: str, dst: Sequence[int], srcs: Sequence[Sequence[Bit]]) -> None:
        values = [_constant(s) for s in srcs]
        if all(v is not None for v in values):  # nothing to compute at run time
            self.load(dst, self.fold(opcode, values))
            return
        getattr(self, f"op_{opcode}", self.unsupported(opcode))(dst, *srcs)

    def fold(self, opcode: str, v: Sequence[int]) -> int:
        mask = (1 << self.w) - 1
        table = {
            "copy": lambda a: a, "add": lambda a, b: (a + b) & mask, "sub": lambda a, b: (a - b) & mask,
            "mul": lambda a, b: (a * b) & mask, "lt": lambda a, b: int(a < b), "gt": lambda a, b: int(a > b),
            "eq": lambda a, b: int(a == b), "ne": lambda a, b: int(a != b),
            "and": lambda a, b: (a & b) & 1, "or": lambda a, b: (a | b) & 1, "not": lambda a: 1 - (a & 1),
        }
        if opcode not in table:
            raise CircuitError(f"{opcode} is not supported yet")
        return table[opcode](*v)

    def unsupported(self, opcode: str):
        def fail(*_):
            raise CircuitError(f"{opcode} is not supported yet")
        return fail

    def op_copy(self, dst, a):
        self.copy(dst, a)

    def op_not(self, dst, a):
        self.gate(dst[0])
        self.gate(dst[0], [(a[0], True)])

    def op_and(self, dst, a, b):
        self.gate(dst[0], [(a[0], True), (b[0], True)])

    def op_or(self, dst, a, b):
        self.gate(dst[0], [(a[0], True)])
        self.gate(dst[0], [(b[0], True)])
        self.gate(dst[0], [(a[0], True), (b[0], True)])

    def op_add(self, dst, a, b):
        if _constant(a) is not None:
            a, b = b, a
        if _constant(b) is not None:
            self.copy(dst, a)
            self.add_constant(dst, _constant(b))
        elif _same(a, b):  # a + a is a shifted by one
            for i in range(1, self.w):
                self.gate(dst[i], [(a[i - 1], True)])
        else:
            self.copy(dst, a)
            self.gates += adder(dst, [bit.index for bit in b], self.scratch)

    def op_sub(self, dst, a, b):
        if _constant(b) is not None:
            self.copy(dst, a)
            self.add_constant(dst, (-_constant(b)) & ((1 << self.w) - 1))
        elif _same(a, b):
            return
        else:
            if _constant(a) is None:
                self.copy(dst, a)
            else:
                self.load(dst, _constant(a))
            self.gates += list(reversed(adder(dst, [bit.index for bit in b], self.scratch)))

    def op_mul(self, dst, a, b):
        if _constant(a) is not None:
            a, b = b, a
        constant = _constant(b)
        if constant is not None:
            for j in range(self.w):
                if (constant >> j) & 1:
                    self.shifted_add(dst, [bit.index for bit in a], j)
        elif _same(a, b):
            raise CircuitError("squaring a register needs a copy of it, which is not supported yet")
        else:
            for j in range(self.w):
                self.shifted_add(dst, [bit.index for bit in a], j, [(b[j], True)])

    def op_eq(self, dst, a, b):
        if _constant(a) is not None:
            a, b = b, a
        constant = _constant(b)
        if constant is not None:
            self.gate(dst[0], [(a[i], bool((constant >> i) & 1)) for i in range(self.w)])
        elif _same(a, b):
            self.gate(dst[0])
        else:
            pairs = [Gate(y.index, ((x.index, True),)) for x, y in zip(a, b, strict=True)]
            self.gates += pairs
            self.gate(dst[0], [(y, False) for y in b])
            self.gates += list(reversed(pairs))

    def op_ne(self, dst, a, b):
        self.op_eq(dst, a, b)
        self.gate(dst[0])

    def op_lt(self, dst, a, b):
        w = self.w
        if _constant(b) is not None:  # a < c: the first bit that differs is a 0 in a where c has a 1
            c = _constant(b)
            for i in range(w):
                if (c >> i) & 1:
                    self.gate(dst[0], [(a[i], False)] + [(a[j], bool((c >> j) & 1)) for j in range(i + 1, w)])
        elif _constant(a) is not None:  # c < b: the first bit that differs is a 1 in b where c has a 0
            c = _constant(a)
            for i in range(w):
                if not (c >> i) & 1:
                    self.gate(dst[0], [(b[i], True)] + [(b[j], bool((c >> j) & 1)) for j in range(i + 1, w)])
        elif not _same(a, b):
            qa, qb = [bit.index for bit in a], [bit.index for bit in b]
            flip = [Gate(q) for q in qa]  # a < b is the carry out of (not a) + b
            chain = self.carry_chain(qa, qb)
            self.gates += flip + chain
            self.gate(dst[0], [(Q(qa[-1]), True)])
            self.gates += list(reversed(chain)) + flip

    def op_gt(self, dst, a, b):
        self.op_lt(dst, b, a)


def _bits(operand: Operand, width: int, needed: int) -> List[Bit]:
    if isinstance(operand, Const):
        value = operand.value
        if isinstance(value, float) and value != int(value):
            raise CircuitError(f"{value} is not an integer")
        value = int(value) & ((1 << width) - 1)
        return [C((value >> i) & 1) for i in range(width)]
    slot = int(operand.name[1:])
    if not operand.name.startswith("R") or slot >= needed:
        raise CircuitError(f"{operand.name} is not an allocated slot")
    return [Q(slot * width + i) for i in range(width)]


def to_openqasm(allocation: Allocation) -> Circuit:
    """The circuit for an allocated program, as OpenQASM 3 text and as a list of gates."""
    width, needed = allocation.register_width, allocation.needed
    scratch = needed * width
    qubits = (needed + 1) * width
    regions: List[Tuple[int, bool]] = []
    gates: List[Gate] = []
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"qubit[{qubits}] q;", ""]

    def emit(produced: Sequence[Gate]) -> None:
        for gate in produced:
            full = Gate(gate.target, gate.controls + tuple(regions))
            gates.append(full)
            lines.append(gate_text(full))

    for instruction in allocation.program.body:
        if isinstance(instruction, (Alloc, Free)):
            continue  # a register's life is not a gate
        if isinstance(instruction, EndCtrl):
            regions.pop()
        elif isinstance(instruction, (Op, IOp)):
            dst = _bits(Reg(instruction.dst), width, needed)
            builder = _Builder(width, scratch)
            builder.op(instruction.opcode, [bit.index for bit in dst], [_bits(s, width, needed) for s in instruction.sources])
            emit(builder.gates if isinstance(instruction, Op) else list(reversed(builder.gates)))
        elif isinstance(instruction, (Ctrl, ICtrl)):
            control = _bits(instruction.control, width, needed)[0]
            if not isinstance(control, Q):
                raise CircuitError("a control must be a register")
            regions.append((control.index, isinstance(instruction, Ctrl)))
        else:
            raise CircuitError(f"cannot turn {type(instruction).__name__} into gates")
    if not isinstance(allocation.program.result, Reg):
        raise CircuitError("a program must return a register; run flatten first")
    result = tuple(_bits(allocation.program.result, width, needed))
    measured = tuple(bit.index for bit in result)
    lines.insert(3, f"bit[{len(measured)}] result;")
    lines += [f"result[{i}] = measure q[{q}];" for i, q in enumerate(measured)]
    return Circuit("\n".join(lines) + "\n", qubits, width, tuple(gates), result, measured)

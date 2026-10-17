"""
Residue number system (RNS) encoding. An integer is held as its remainders after division by a few small moduli that share
no factor, each remainder in its own small binary register (a "channel"). The moduli are 2^n - 1, 2^n and 2^n + 1, with the
smallest n whose product covers 2^width values, so the width decides how many qubits each channel has and the range (the
product, read as a signed number, in place of -2^(w-1) to 2^(w-1)-1).

Addition, subtraction and multiplication act on each channel on its own, with no carry from one channel to the next, so the
channels have no gate between them and run in the same layer. Equality needs no more: two numbers are equal when every remainder
matches. What the remainders do not show is magnitude, so `<` and `>` have no circuit here (they need a conversion to a form that
orders numbers, such as mixed-radix digits) and are refused, as is anything built from them: a branch on a comparison.

Adding a register or a constant to a channel is the standard modular adder: add, take away the modulus, copy the sign, put the
modulus back if it was negative, then clear the flag by comparing the result with the number added. A channel whose modulus is
a power of two needs only the ordinary binary adder, whose wrap is the modulus. Every channel has three scratch qubits of its
own (carry, overflow, flag) so that no two channels share a wire. Multiplying adds, for each pair of bits of the operands, the
constant 2^(i+j) modulo the modulus, controlled on both bits.

A boolean is the number 1 or 0, so it is held as remainders like any other value, and read as a condition from the first bit.
The returned register is measured whole, and the readout recipe carries the moduli so the remainders are combined into the
number classically (one shot, exact).

The comparison page and the tests treat an RNS result as right when it equals what the program returns without wrapping, taken
into this range, because the range is not 2^width.
"""
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from compiler.compiler import CIRCUIT, ISA, compile_pass
from compiler.parser.instructions import (
    Alloc, Const, Ctrl, EndCtrl, Free, ICtrl, IOp, IsaProgram, Op, Reg, Scratch, Update,
)
from compiler.passes.qasm import C, CircuitError, Q, _Builder
from compiler.system import System
from experiments.encodings.readout import Readout

Channels = List[List[int]]  # the qubits of each channel of a register, least significant bit first
Value = Union[int, Channels]  # a constant, or a register


def moduli_for(width: int) -> Tuple[int, int, int]:
    """2^n - 1, 2^n and 2^n + 1 for the smallest n (at least 2) whose product covers 2^width values."""
    n = 2
    while (1 << n) * ((1 << (2 * n)) - 1) < (1 << width):
        n += 1
    return (1 << n) - 1, 1 << n, (1 << n) + 1


def modulus_of(system: System) -> int:
    """How many values a register holds on this system: the product of its moduli."""
    return math.prod(moduli_for(system.register_width))


@dataclass(frozen=True)
class Layout:
    moduli: Tuple[int, ...]

    @property
    def bits(self) -> Tuple[int, ...]:
        return tuple((m - 1).bit_length() for m in self.moduli)  # a remainder's largest value is m - 1

    @property
    def size(self) -> int:
        return sum(self.bits)

    def channels(self, slot: int) -> Channels:
        start, result = slot * self.size, []
        for size in self.bits:
            result.append(list(range(start, start + size)))
            start += size
        return result


class _RnsBuilder(_Builder):
    """The gates of one instruction. `on(i)` points the builder's width and carry qubit at channel i."""

    def __init__(self, circuit: QuantumCircuit, layout: Layout, scratch: int, regions: Sequence[Tuple[int, bool]]) -> None:
        super().__init__(circuit, layout.size, scratch, regions)
        self.layout = layout
        self.base = scratch

    @contextmanager
    def on(self, i: int):
        before = (self.w, self.scratch)
        self.w, self.scratch = self.layout.bits[i], self.base + 3 * i
        try:
            yield
        finally:
            self.w, self.scratch = before

    def over(self, i: int) -> int:
        return self.base + 3 * i + 1

    def flag(self, i: int) -> int:
        return self.base + 3 * i + 2

    # -- one channel ---------------------------------------------------------------------------------------------

    def adder_carry_out(self, acc: Sequence[int], add: Sequence[int], carry: int, out: int) -> None:
        """acc += add, with the carry out of the top bit written to `out`; `add` and `carry` are restored."""
        n = len(acc)
        carries = [carry] + list(add[:n - 1])
        for i in range(n):
            self.majority(carries[i], acc[i], add[i])
        self.x(out, [(add[n - 1], True)])
        for i in reversed(range(n)):
            self.unmajority(carries[i], acc[i], add[i])

    def clear_flag(self, i: int, acc: Sequence[int], other: Sequence) -> None:
        """The flag is set exactly when the sum was below the modulus, which is when the result is not below what was added."""
        self.lt_unsigned([self.flag(i)], [Q(q) for q in acc], list(other))
        self.x(self.flag(i))

    def correct(self, i: int, acc: Sequence[int]) -> None:
        """For acc + (what was added) in k+1 bits: take away the modulus, copy the sign, put it back if it was negative."""
        m, k = self.layout.moduli[i], self.layout.bits[i]
        extended = list(acc) + [self.over(i)]
        self.add_constant(extended, (1 << (k + 1)) - m)
        self.x(self.flag(i), [(self.over(i), True)])
        with self.controlled_on([(Q(self.flag(i)), True)]):
            self.add_constant(extended, m)

    def add_const(self, i: int, acc: Sequence[int], c: int) -> None:
        """acc += c (modulo the modulus of channel i)."""
        m, k = self.layout.moduli[i], self.layout.bits[i]
        c %= m
        if c == 0:
            return
        with self.on(i):
            if m == 1 << k:
                return self.add_constant(acc, c)
            self.add_constant(list(acc) + [self.over(i)], c)
            self.correct(i, acc)
            self.clear_flag(i, acc, [C((c >> j) & 1) for j in range(k)])

    def add_reg(self, i: int, acc: Sequence[int], add: Sequence[int]) -> None:
        """acc += add, both registers of channel i holding numbers below its modulus."""
        m, k = self.layout.moduli[i], self.layout.bits[i]
        with self.on(i):
            if m == 1 << k:
                return self.adder(acc, add, self.scratch)
            self.adder_carry_out(acc, add, self.scratch, self.over(i))
            self.correct(i, acc)
            self.clear_flag(i, acc, [Q(q) for q in add])

    # -- whole registers -----------------------------------------------------------------------------------------

    def residues(self, value: int) -> List[int]:
        return [value % m for m in self.layout.moduli]

    def load_value(self, dst: Channels, value: int) -> None:
        for channel, residue in zip(dst, self.residues(value), strict=True):
            self.load(channel, residue)

    def copy_register(self, dst: Channels, src: Channels) -> None:
        for d, s in zip(dst, src, strict=True):
            for dq, sq in zip(d, s, strict=True):
                self.x(dq, [(sq, True)])

    def bits_of(self, value: Value) -> List:
        """Every bit of a value, channel after channel: wires for a register and constants for a number."""
        if isinstance(value, int):
            return [C((r >> j) & 1) for r, size in zip(self.residues(value), self.layout.bits) for j in range(size)]
        return [Q(q) for channel in value for q in channel]

    def condition_bit(self, value: Value) -> List:
        """A boolean is read from the first bit of the register."""
        return [C(value & 1)] if isinstance(value, int) else [Q(value[0][0])]

    def boolean(self, dst: Channels, build) -> None:
        """Run `build` into the first bit of `dst`, then copy it to the first bit of every other channel: 1 is 1 in every channel."""
        first = dst[0][0]
        build([first])
        for channel in dst[1:]:
            self.x(channel[0], [(first, True)])

    def fold(self, opcode: str, values: Sequence[int]) -> int:
        a, b = (list(values) + [0])[:2]
        table = {
            "copy": a, "add": a + b, "sub": a - b, "mul": a * b, "lt": int(a < b), "gt": int(a > b),
            "eq": int(a == b), "ne": int(a != b), "and": (a & b) & 1, "or": (a | b) & 1, "not": 1 - (a & 1),
        }
        if opcode not in table:
            raise CircuitError(f"{opcode} is not supported yet")
        return table[opcode]

    def op(self, opcode: str, dst: Channels, srcs: Sequence[Value]) -> None:
        if all(isinstance(s, int) for s in srcs):  # nothing to compute at run time
            return self.load_value(dst, self.fold(opcode, srcs))
        if opcode in ("lt", "gt"):
            raise CircuitError(
                f"{opcode} on registers has no circuit in the residue number system: ordering needs the magnitude of a number, "
                "which its remainders do not show (a conversion to mixed-radix digits would give it)")
        handler = getattr(self, f"rns_{opcode}", None)
        if handler is None:
            raise CircuitError(f"{opcode} is not supported yet")
        handler(dst, *srcs)

    def rns_copy(self, dst: Channels, a: Value) -> None:
        self.copy_register(dst, a)

    def accumulate(self, dst: Channels, source: Value, sign: int) -> None:
        """dst += sign * source, a constant or a register, channel by channel."""
        for i, channel in enumerate(dst):
            if isinstance(source, int):
                self.add_const(i, channel, sign * source)
            elif sign > 0:
                self.add_reg(i, channel, source[i])
            else:
                with self.undone():
                    self.add_reg(i, channel, source[i])

    def rns_add(self, dst: Channels, a: Value, b: Value) -> None:
        if isinstance(a, int):
            a, b = b, a
        self.copy_register(dst, a)
        self.accumulate(dst, b, 1)

    def rns_sub(self, dst: Channels, a: Value, b: Value) -> None:
        if not isinstance(a, int) and not isinstance(b, int) and a == b:  # a - a is zero
            return
        if isinstance(a, int):
            self.load_value(dst, a)
        else:
            self.copy_register(dst, a)
        self.accumulate(dst, b, -1)

    def rns_mul(self, dst: Channels, a: Value, b: Value) -> None:
        if isinstance(a, int):
            a, b = b, a
        if not isinstance(b, int) and a == b:
            raise CircuitError("squaring a register needs a copy of it, which is not supported yet")
        for i, channel in enumerate(dst):
            m = self.layout.moduli[i]
            for ia, qa in enumerate(a[i]):
                if isinstance(b, int):
                    with self.controlled_on([(Q(qa), True)]):
                        self.add_const(i, channel, (b * (1 << ia)) % m)
                else:
                    for ib, qb in enumerate(b[i]):
                        if (1 << (ia + ib)) % m:
                            with self.controlled_on([(Q(qa), True), (Q(qb), True)]):
                                self.add_const(i, channel, (1 << (ia + ib)) % m)

    def rns_eq(self, dst: Channels, a: Value, b: Value) -> None:
        with self.on_all():
            self.boolean(dst, lambda d: self.op_eq(d, self.bits_of(a), self.bits_of(b)))

    def rns_ne(self, dst: Channels, a: Value, b: Value) -> None:
        with self.on_all():
            self.boolean(dst, lambda d: self.op_ne(d, self.bits_of(a), self.bits_of(b)))

    def rns_and(self, dst: Channels, a: Value, b: Value) -> None:
        self.boolean(dst, lambda d: self.op_and(d, self.condition_bit(a), self.condition_bit(b)))

    def rns_or(self, dst: Channels, a: Value, b: Value) -> None:
        self.boolean(dst, lambda d: self.op_or(d, self.condition_bit(a), self.condition_bit(b)))

    def rns_not(self, dst: Channels, a: Value) -> None:
        self.boolean(dst, lambda d: self.op_not(d, self.condition_bit(a)))

    @contextmanager
    def on_all(self):
        """Point the builder's width at a whole register, for the circuits that treat all its bits alike."""
        before = self.w
        self.w = self.layout.size
        try:
            yield
        finally:
            self.w = before

    def update(self, opcode: str, dst: Channels, src: Value) -> None:
        """dst += src or dst -= src in place."""
        if not isinstance(src, int) and src == dst:
            raise CircuitError("a register cannot be updated with itself: the result cannot be undone")
        self.accumulate(dst, src, 1 if opcode == "add" else -1)


@compile_pass(
    ISA, CIRCUIT,
    title="Lower to a residue number system circuit",
    summary=(
        "A number held as its remainders by 2^n-1, 2^n and 2^n+1, each in a small binary register. Add, subtract and "
        "multiply act on each remainder on its own with no carry between them; equality compares remainders. "
        "Ordering has no circuit, so comparisons are refused."
    ),
)
def to_circuit_rns(program: IsaProgram, system: Optional[System] = None) -> QuantumCircuit:
    if system is None:
        raise CircuitError("the lowering needs the system the program was allocated for")
    layout = Layout(moduli_for(system.register_width))
    scratch_register = next((i.register for i in program.body if isinstance(i, Scratch)), None)
    if scratch_register is None:
        raise CircuitError("the program declares no scratch register; run allocate first")
    needed = int(scratch_register[1:])
    scratch = needed * layout.size
    circuit = QuantumCircuit(*[QuantumRegister(layout.size, f"R{i}") for i in range(needed)],
                             QuantumRegister(3 * len(layout.moduli), "scratch"))
    regions: List[Tuple[int, bool]] = []

    def builder() -> _RnsBuilder:
        return _RnsBuilder(circuit, layout, scratch, regions)

    def register(name: str) -> Channels:
        slot = int(name[1:])
        if not name.startswith("R") or slot >= needed:
            raise CircuitError(f"{name} is not an allocated slot")
        return layout.channels(slot)

    def value(operand) -> Value:
        if isinstance(operand, Const):
            number = operand.value
            if isinstance(number, float) and number != int(number):
                raise CircuitError(f"{number} is not an integer")
            return int(number)
        return register(operand.name)

    for instruction in program.body:
        if isinstance(instruction, (Alloc, Free, Scratch)):
            continue
        if isinstance(instruction, EndCtrl):
            regions.pop()
        elif isinstance(instruction, Op):
            builder().op(instruction.opcode, register(instruction.dst), [value(s) for s in instruction.sources])
        elif isinstance(instruction, IOp):
            undo = builder()
            with undo.undone():
                undo.op(instruction.opcode, register(instruction.dst), [value(s) for s in instruction.sources])
        elif isinstance(instruction, Update):
            builder().update(instruction.opcode, register(instruction.dst), value(instruction.source))
        elif isinstance(instruction, (Ctrl, ICtrl)):
            control = value(instruction.control)
            if isinstance(control, int):
                raise CircuitError("a control must be a register")
            regions.append((control[0][0], isinstance(instruction, Ctrl)))
        else:
            raise CircuitError(f"cannot turn {type(instruction).__name__} into gates")
    circuit.metadata = {}
    if program.result is None:  # just the instructions: nothing to read out
        return circuit
    if not isinstance(program.result, Reg):
        raise CircuitError("a program must return a register; run flatten first")
    result = [q for channel in register(program.result.name) for q in channel]
    circuit.add_register(ClassicalRegister(len(result), "result"))
    for bit, qubit in enumerate(result):
        circuit.measure(circuit.qubits[qubit], circuit.clbits[bit])
    return Readout(shots=1, decode="rns:" + ",".join(str(m) for m in layout.moduli)).attach(circuit)

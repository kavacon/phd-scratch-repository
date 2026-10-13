"""
Circuit generation: turn an allocated instruction program into a circuit (a Qiskit `QuantumCircuit`, written out as OpenQASM 3).

Layout. Slot `i` is the qubits `i*w` to `i*w + w - 1`, least significant bit first, where `w` is the system's register
width. The scratch register the program declares with `SCRATCH` follows the slots, so the circuit has `(slots used + 1) * w` qubits.
An integer is unsigned and arithmetic is modulo 2^w. A boolean (a comparison, `and`, `or`, `not`) is bit 0 of its
register with the other bits zero; `CTRL` and the logical operations read only bit 0.

Every gate here is an X with some controls (X, CNOT, Toffoli, multi-controlled X), each control positive or
negative. They are appended straight to a Qiskit `QuantumCircuit`, which is the circuit every later stage works on;
one register per slot (`R0`, `R1`, ...) and one for scratch. Each opcode writes straight into its destination, which starts at
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
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qasm3
from qiskit.circuit.library import CCXGate, CXGate, MCXGate, XGate

from compiler.parser.instructions import (
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
    IsaProgram,
    Scratch,
)
from compiler.compiler import CIRCUIT, ISA, compile_pass
from compiler.system import System


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
    """An X gate as read back out of a circuit: X on `target`, applied when every control holds (a control is (qubit, polarity))."""
    target: int
    controls: Tuple[Tuple[int, bool], ...] = ()


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


def _x_gate(controls: Sequence[Tuple[int, bool]]):
    """Qiskit's gate for an X with these controls (a control is (qubit, polarity))."""
    count = len(controls)
    state = sum(int(polarity) << i for i, (_, polarity) in enumerate(controls))
    if count == 0:
        return XGate()
    if count == 1:
        return CXGate(ctrl_state=state)
    if count == 2:
        return CCXGate(ctrl_state=state)
    return MCXGate(count, ctrl_state=state)


def _distinct(controls: Sequence[Tuple[int, bool]]) -> Optional[Tuple[Tuple[int, bool], ...]]:
    """The controls with a qubit named twice kept once, or None if it is asked to be both set and clear (the gate never fires)."""
    seen: dict = {}
    for qubit, polarity in controls:
        if seen.setdefault(qubit, polarity) != polarity:
            return None
    return tuple(seen.items())


def _same(a: Sequence[Bit], b: Sequence[Bit]) -> bool:
    return all(isinstance(x, Q) for x in a) and list(a) == list(b)


def _constant(bits: Sequence[Bit]) -> Optional[int]:
    if all(isinstance(b, C) for b in bits):
        return sum(b.value << i for i, b in enumerate(bits))
    return None


class _Builder:
    """
    The gates of one instruction, appended straight to a Qiskit circuit. Every gate also carries `regions` (the
    controls of the `CTRL` regions the instruction sits in) and whatever extra controls are in force. A block
    is undone by building it inside `undone()`: every gate here is its own inverse, so the inverse is the same
    gates in reverse order, which is what `QuantumCircuit.inverse` gives.
    """

    def __init__(self, circuit: QuantumCircuit, width: int, scratch: int, regions: Sequence[Tuple[int, bool]] = ()) -> None:
        self.circuit = circuit
        self.w = width
        self.scratch = scratch
        self.regions = tuple(regions)
        self.extra: Tuple[Tuple[int, bool], ...] = ()

    @contextmanager
    def undone(self):
        """Everything built inside is added to the circuit as its inverse."""
        outer = self.circuit
        self.circuit = QuantumCircuit(*outer.qregs)
        try:
            yield
        finally:
            inner, self.circuit = self.circuit, outer
            outer.compose(inner.inverse(), inplace=True)

    @contextmanager
    def controlled_on(self, controls: Sequence[Tuple[Bit, bool]]):
        """Every gate built inside also needs these controls (a constant one that cannot hold removes the gate)."""
        before = self.extra
        self.extra = before + tuple((bit.index, p) for bit, p in controls)
        try:
            yield
        finally:
            self.extra = before

    def x(self, target: int, controls: Sequence[Tuple[int, bool]] = ()) -> None:
        """An X on `target` with controls on wires (a control is (qubit, polarity))."""
        kept = _distinct(tuple(controls) + self.extra + self.regions)
        if kept is None:  # asked to be both set and clear: the gate can never fire
            return
        self.circuit.append(_x_gate(kept), [self.circuit.qubits[q] for q, _ in kept] + [self.circuit.qubits[target]])

    def gate(self, target: int, controls: Sequence[Tuple[Bit, bool]] = ()) -> None:
        kept = []
        for bit, polarity in controls:
            if isinstance(bit, C):
                if bool(bit.value) != polarity:
                    return  # a control that can never hold, so the gate does nothing
            else:
                kept.append((bit.index, polarity))
        self.x(target, kept)

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

    # -- the adder -----------------------------------------------------------------------------------------------

    def majority(self, c: int, b: int, a: int) -> None:
        self.x(b, [(a, True)])
        self.x(c, [(a, True)])
        self.x(a, [(c, True), (b, True)])

    def unmajority(self, c: int, b: int, a: int) -> None:
        self.x(a, [(c, True), (b, True)])
        self.x(c, [(a, True)])
        self.x(b, [(c, True)])

    def adder(self, acc: Sequence[int], add: Sequence[int], carry: int) -> None:
        """acc += add (modulo 2^n) with a Cuccaro adder; `add` and `carry` are restored."""
        n = len(acc)
        if n == 0:
            return
        if n == 1:
            self.x(acc[0], [(add[0], True)])
            return
        carries = [carry] + list(add[:n - 1])
        for i in range(n - 1):
            self.majority(carries[i], acc[i], add[i])
        self.x(acc[n - 1], [(add[n - 1], True)])
        self.x(acc[n - 1], [(add[n - 2], True)])
        for i in reversed(range(n - 1)):
            self.unmajority(carries[i], acc[i], add[i])

    def shifted_add(self, dst: Sequence[int], src: Sequence[int], shift: int, controls: Sequence[Tuple[Bit, bool]] = ()) -> None:
        with self.controlled_on(controls):
            self.adder(dst[shift:], src[:self.w - shift], self.scratch)

    def carry_chain(self, a: Sequence[int], b: Sequence[int]) -> None:
        """After these gates the carry out of a + b sits where a's top bit was; they are undone by building them in `undone()`."""
        carries = [self.scratch] + list(a[:-1])
        for i in range(len(a)):
            self.majority(carries[i], b[i], a[i])

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
            self.adder(dst, [bit.index for bit in b], self.scratch)

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
            with self.undone():  # subtraction is the adder run backwards
                self.adder(dst, [bit.index for bit in b], self.scratch)

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
            def xor() -> None:  # b ^= a, undone after the test
                for x, y in zip(a, b, strict=True):
                    self.x(y.index, [(x.index, True)])
            xor()
            self.gate(dst[0], [(y, False) for y in b])
            with self.undone():
                xor()

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

            def borrow() -> None:  # a < b is the carry out of (not a) + b
                for q in qa:
                    self.x(q)
                self.carry_chain(qa, qb)
            borrow()
            self.gate(dst[0], [(Q(qa[-1]), True)])
            with self.undone():
                borrow()

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


def _is_x(operation) -> bool:
    base = getattr(operation, "base_gate", None)
    return operation.name == "x" or (base is not None and base.name == "x")


def x_gates(circuit: QuantumCircuit) -> List[Gate]:
    """
    The X gates of a circuit this module generated, as (target, controls) in order, so they can be run on bits or
    printed. Measurements and barriers are left out. Any other gate is an error.
    """
    gates: List[Gate] = []
    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name in ("measure", "barrier"):
            continue
        if not _is_x(operation):
            raise CircuitError(f"{operation.name} is not an X gate")
        wires = [circuit.find_bit(q).index for q in instruction.qubits]
        count = getattr(operation, "num_ctrl_qubits", 0)
        state = getattr(operation, "ctrl_state", (1 << count) - 1)
        gates.append(Gate(wires[-1], tuple((wires[i], bool((state >> i) & 1)) for i in range(count))))
    return gates


def measured_qubits(circuit: QuantumCircuit) -> List[int]:
    """The qubits measured at the end, in the order of the classical bits they are written to."""
    pairs = []
    for instruction in circuit.data:
        if instruction.operation.name == "measure":
            pairs.append((circuit.find_bit(instruction.clbits[0]).index, circuit.find_bit(instruction.qubits[0]).index))
    return [qubit for _, qubit in sorted(pairs)]


def openqasm_text(circuit: QuantumCircuit) -> str:
    """
    The circuit as OpenQASM 3: one register of qubits `q`, X gates with `ctrl` and `negctrl` modifiers, and a
    measurement per returned bit. A circuit holding other gates (Qiskit's rewrite into `u` and `cx`) is written by Qiskit.
    """
    if any(not (_is_x(i.operation) or i.operation.name in ("measure", "barrier")) for i in circuit.data):
        return qasm3.dumps(circuit)
    measured = measured_qubits(circuit)
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"qubit[{circuit.num_qubits}] q;"]
    if measured:
        lines.append(f"bit[{len(measured)}] result;")
    lines.append("")
    lines += [gate_text(gate) for gate in x_gates(circuit)]
    lines += [f"result[{i}] = measure q[{q}];" for i, q in enumerate(measured)]
    return "\n".join(lines) + "\n"


@compile_pass(
    ISA, CIRCUIT,
    title="Generate OpenQASM",
    summary=(
        "Writes each instruction as X gates with controls on a Qiskit circuit (a register per slot, plus "
        "scratch) and measures the returned register."
    ),
)
def to_openqasm(program: IsaProgram, system: Optional[System] = None) -> QuantumCircuit:
    """
    The circuit for an allocated program: a Qiskit circuit with a register per slot and one for scratch. `system` is
    the machine the program was allocated for; it says how many qubits make a register.
    """
    if system is None:
        raise CircuitError("to_openqasm needs the system the program was allocated for")
    width = system.register_width
    scratch_register = next((i.register for i in program.body if isinstance(i, Scratch)), None)
    if scratch_register is None:
        raise CircuitError("the program declares no scratch register; run allocate first")
    needed = int(scratch_register[1:])  # the slots come first and the scratch register follows them
    scratch = needed * width
    circuit = QuantumCircuit(*[QuantumRegister(width, f"R{i}") for i in range(needed)], QuantumRegister(width, "scratch"))
    regions: List[Tuple[int, bool]] = []

    for instruction in program.body:
        if isinstance(instruction, (Alloc, Free, Scratch)):
            continue  # a register's life is not a gate
        if isinstance(instruction, EndCtrl):
            regions.pop()
        elif isinstance(instruction, (Op, IOp)):
            dst = [bit.index for bit in _bits(Reg(instruction.dst), width, needed)]
            sources = [_bits(source, width, needed) for source in instruction.sources]
            target = circuit if isinstance(instruction, Op) else QuantumCircuit(*circuit.qregs)
            _Builder(target, width, scratch, regions).op(instruction.opcode, dst, sources)
            if isinstance(instruction, IOp):  # the inverse instruction is the same gates in reverse order
                circuit.compose(target.inverse(), inplace=True)
        elif isinstance(instruction, (Ctrl, ICtrl)):
            control = _bits(instruction.control, width, needed)[0]
            if not isinstance(control, Q):
                raise CircuitError("a control must be a register")
            regions.append((control.index, isinstance(instruction, Ctrl)))
        else:
            raise CircuitError(f"cannot turn {type(instruction).__name__} into gates")
    if not isinstance(program.result, Reg):
        raise CircuitError("a program must return a register; run flatten first")
    result = _bits(program.result, width, needed)
    circuit.add_register(ClassicalRegister(len(result), "result"))
    for i, bit in enumerate(result):
        circuit.measure(circuit.qubits[bit.index], circuit.clbits[i])
    return circuit

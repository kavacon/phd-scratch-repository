"""
Fourier (phase) encoding: an integer is held as phases across its n qubits while it is being added into, and as plain
bits whenever anything reads it. Arithmetic is modulo 2^n, so two's complement and wrapping come for free.

Adding into a register in the Fourier basis is a rotation per qubit. A constant is plain rotations. Adding another
register rotates by the bits of that register, one controlled rotation per bit and qubit, so the register being read
has to be plain bits: a control qubit must hold a 0 or a 1. Comparisons, equality, the Boolean operations, branch
conditions and the returned value read bits too.

The lowering tracks which basis each register is in as it goes through the instructions, and converts only when an
instruction needs the other one. A register converted to plain bits for a comparison stays plain until something adds
into it again. Conversions are never controlled: a basis change is a fact about the circuit, not about the branch it
sits in, so a controlled region needs no join rule. A register that has just been allocated is zero, and the transform
of zero is a Hadamard on each qubit, so that is all an arithmetic instruction writes for its destination.

The transform is written out as gates (no swaps, so the Fourier register is bit-reversed: qubit p carries the phase
2*pi*x / 2^(p+1)). Comparisons, equality and Booleans are the shared binary circuits on plain bits.

This is a local copy of the shared lowering's plumbing with arithmetic replaced; the shared pass is untouched.
"""
import math
from typing import List, Optional, Sequence, Set, Tuple

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import CPhaseGate, PhaseGate

from compiler.compiler import CIRCUIT, ISA, compile_pass
from compiler.parser.instructions import (
    Alloc, Ctrl, EndCtrl, Free, ICtrl, IOp, IsaProgram, Op, Reg, Scratch, Update,
)
from compiler.passes.qasm import CircuitError, Q, _bits, _constant, _Builder, _distinct, _same
from compiler.system import System
from experiments.encodings.readout import Readout

ARITHMETIC = ("add", "sub", "mul")


def _transform(qubits: Sequence[int]) -> List[Tuple]:
    """The gates of the transform, in order: ("h", q) and ("cp", angle, control, target). The register ends bit-reversed."""
    n = len(qubits)
    gates: List[Tuple] = []
    for i in reversed(range(n)):
        gates.append(("h", qubits[i]))
        for j in reversed(range(i)):
            gates.append(("cp", math.pi / 2 ** (i - j), qubits[j], qubits[i]))
    return gates


class _FourierBuilder(_Builder):
    """The gates of one instruction. `basis` holds the slots currently in the Fourier basis, and is shared by every instruction."""

    def __init__(self, circuit: QuantumCircuit, width: int, scratch: int, regions: Sequence[Tuple[int, bool]], basis: Set[int]) -> None:
        super().__init__(circuit, width, scratch, regions)
        self.basis = basis

    # -- basis changes, which are never controlled ---------------------------------------------------------------

    def slot(self, qubits: Sequence[int]) -> int:
        return qubits[0] // self.w

    def apply(self, gates: Sequence[Tuple], inverse: bool) -> None:
        for gate in (reversed(gates) if inverse else gates):
            if gate[0] == "h":
                self.circuit.h(gate[1])
            else:
                _, angle, control, target = gate
                self.circuit.append(CPhaseGate(-angle if inverse else angle), [self.circuit.qubits[control], self.circuit.qubits[target]])

    def to_fourier(self, qubits: Sequence[int]) -> None:
        if self.slot(qubits) not in self.basis:
            self.apply(_transform(qubits), inverse=False)
            self.basis.add(self.slot(qubits))

    def to_plain(self, qubits: Sequence[int]) -> None:
        if self.slot(qubits) in self.basis:
            self.apply(_transform(qubits), inverse=True)
            self.basis.discard(self.slot(qubits))

    def plain_sources(self, sources) -> None:
        for source in sources:
            if _constant(source) is None:
                self.to_plain([bit.index for bit in source])

    def start_from_zero(self, dst: Sequence[int]) -> None:
        """
        The destination of an arithmetic instruction that always runs is zero, and the transform of zero is a Hadamard
        on each qubit. In a controlled region it may hold the value the other branch wrote (both branches of an
        if/else write the same register), so it takes the full transform, which is right whatever it holds.
        """
        if self.regions:
            return self.to_fourier(dst)
        for q in dst:
            self.circuit.h(q)
        self.basis.add(self.slot(dst))

    def end_at_zero(self, dst: Sequence[int]) -> None:
        if self.regions:
            return self.to_plain(dst)
        for q in dst:
            self.circuit.h(q)
        self.basis.discard(self.slot(dst))

    # -- rotations -----------------------------------------------------------------------------------------------

    def phase(self, target: int, angle: float, controls: Sequence[Tuple[int, bool]]) -> None:
        """A phase rotation on `target`, with the controls given plus those of the regions the instruction sits in."""
        kept = _distinct(tuple(controls) + self.extra + self.regions)
        if kept is None:
            return
        gate = PhaseGate(angle)
        if kept:
            gate = gate.control(len(kept), ctrl_state=sum(int(p) << i for i, (_, p) in enumerate(kept)))
        self.circuit.append(gate, [self.circuit.qubits[q] for q, _ in kept] + [self.circuit.qubits[target]])

    def rotate(self, acc: Sequence[int], value: int, controls: Sequence[Tuple[int, bool]] = ()) -> None:
        """Add `value` (modulo 2^n) into a register in the Fourier basis, when every control holds."""
        for p, q in enumerate(acc):
            period = 2 ** (p + 1)
            if value % period:
                self.phase(q, 2 * math.pi * (value % period) / period, controls)

    def accumulate(self, acc: Sequence[int], source, sign: int) -> None:
        """acc += sign * source, for a constant or a register that is plain bits."""
        constant = _constant(source)
        if constant is not None:
            self.rotate(acc, sign * constant)
        else:
            for m, bit in enumerate(source):
                self.rotate(acc, sign * 2 ** m, [(bit.index, True)])

    # -- arithmetic ----------------------------------------------------------------------------------------------

    def arithmetic(self, opcode: str, dst: Sequence[int], srcs) -> None:
        """The rotations of add, sub and mul into a destination already in the Fourier basis."""
        a, b = srcs
        if opcode == "add":
            self.accumulate(dst, a, 1)
            self.accumulate(dst, b, 1)
        elif opcode == "sub":
            self.accumulate(dst, a, 1)
            self.accumulate(dst, b, -1)
        else:
            if _constant(a) is not None:
                a, b = b, a
            constant = _constant(b)
            if constant is not None:
                for j in range(self.w):
                    if (constant >> j) & 1:
                        for i, bit in enumerate(a):
                            self.rotate(dst, 2 ** (i + j), [(bit.index, True)])
            else:
                for j, multiplier in enumerate(b):
                    for i, bit in enumerate(a):
                        self.rotate(dst, 2 ** (i + j), [(bit.index, True), (multiplier.index, True)])

    def nothing_to_compute(self, opcode: str, srcs) -> bool:
        return opcode == "sub" and _same(*srcs)  # a - a is zero: the destination stays as it is

    def op(self, opcode: str, dst: Sequence[int], srcs) -> None:
        if all(_constant(s) is not None for s in srcs):
            self.to_plain(dst)
            return super().op(opcode, dst, srcs)
        if opcode in ARITHMETIC:
            if opcode == "mul" and _same(*srcs):
                raise CircuitError("squaring a register needs a copy of it, which is not supported yet")
            if self.nothing_to_compute(opcode, srcs):
                return
            self.plain_sources(srcs)
            self.start_from_zero(dst)
            self.arithmetic(opcode, dst, srcs)
        else:
            self.plain_sources(srcs)
            self.to_plain(dst)
            super().op(opcode, dst, srcs)

    def unop(self, opcode: str, dst: Sequence[int], srcs) -> None:
        """The inverse of `op`, for an IOp: the same gates undone, bringing the destination back to zero."""
        if all(_constant(s) is not None for s in srcs) or opcode not in ARITHMETIC:
            self.plain_sources(srcs)
            self.to_plain(dst)
            with self.undone():
                super().op(opcode, dst, srcs)
        elif not self.nothing_to_compute(opcode, srcs):
            self.plain_sources(srcs)
            self.to_fourier(dst)
            with self.undone():
                self.arithmetic(opcode, dst, srcs)
            self.end_at_zero(dst)

    def update(self, opcode: str, dst: Sequence[int], src) -> None:
        """dst += src or dst -= src in place: rotations on a destination in the Fourier basis."""
        constant = _constant(src)
        if constant is None:
            if list(src) == [Q(q) for q in dst]:
                raise CircuitError("a register cannot be updated with itself: the result cannot be undone")
            self.plain_sources([src])
        self.to_fourier(dst)
        self.accumulate(dst, src, 1 if opcode == "add" else -1)


@compile_pass(
    ISA, CIRCUIT,
    title="Lower to a Fourier-encoded circuit",
    summary=(
        "Integers are phases while being added into and plain bits when read. Adding a constant is a rotation per "
        "qubit; adding a register is a rotation controlled by each of its bits. Registers convert only when an "
        "instruction needs the other basis."
    ),
)
def to_circuit_fourier(program: IsaProgram, system: Optional[System] = None) -> QuantumCircuit:
    if system is None:
        raise CircuitError("the lowering needs the system the program was allocated for")
    width = system.register_width
    scratch_register = next((i.register for i in program.body if isinstance(i, Scratch)), None)
    if scratch_register is None:
        raise CircuitError("the program declares no scratch register; run allocate first")
    needed = int(scratch_register[1:])
    scratch = needed * width
    circuit = QuantumCircuit(*[QuantumRegister(width, f"R{i}") for i in range(needed)], QuantumRegister(width, "scratch"))
    regions: List[Tuple[int, bool]] = []
    basis: Set[int] = set()

    def builder() -> _FourierBuilder:
        return _FourierBuilder(circuit, width, scratch, regions, basis)

    def qubits(register: Reg) -> List[int]:
        return [bit.index for bit in _bits(register, width, needed)]

    for instruction in program.body:
        if isinstance(instruction, (Alloc, Scratch)):
            continue
        if isinstance(instruction, Free):
            builder().to_plain(qubits(Reg(instruction.register)))  # a register is reused as plain zero
        elif isinstance(instruction, EndCtrl):
            regions.pop()
        elif isinstance(instruction, Op):
            builder().op(instruction.opcode, qubits(Reg(instruction.dst)), [_bits(s, width, needed) for s in instruction.sources])
        elif isinstance(instruction, IOp):
            builder().unop(instruction.opcode, qubits(Reg(instruction.dst)), [_bits(s, width, needed) for s in instruction.sources])
        elif isinstance(instruction, Update):
            builder().update(instruction.opcode, qubits(Reg(instruction.dst)), _bits(instruction.source, width, needed))
        elif isinstance(instruction, (Ctrl, ICtrl)):
            control = _bits(instruction.control, width, needed)[0]
            if not isinstance(control, Q):
                raise CircuitError("a control must be a register")
            builder().to_plain(qubits(instruction.control))
            regions.append((control.index, isinstance(instruction, Ctrl)))
        else:
            raise CircuitError(f"cannot turn {type(instruction).__name__} into gates")
    if program.result is None:  # just the instructions: nothing to read out, so nothing is converted back
        return circuit
    if not isinstance(program.result, Reg):
        raise CircuitError("a program must return a register; run flatten first")
    builder().to_plain(qubits(program.result))
    result = _bits(program.result, width, needed)
    circuit.add_register(ClassicalRegister(len(result), "result"))
    for i, bit in enumerate(result):
        circuit.measure(circuit.qubits[bit.index], circuit.clbits[i])
    return Readout(shots=1, decode="signed-bits").attach(circuit)

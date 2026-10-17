"""
Fourier (phase) encoding with the registers kept in the Fourier basis and added by the ordinary binary adder.

Take two registers holding a and c as Fourier states. To add a into c, run the binary *subtractor* on the qubits with a's
register as the accumulator and c's register as the addend. The state of a's register comes out as it went in, and
c's register comes out holding a + c: its qubits are the controls of the adder, and the change of value reaches them as
a phase kicked back. The roles are swapped: the register whose value changes is the one that controls the gates. The
reason is that the binary adder written in the Fourier basis is the subtractor with its two registers exchanged. The
registers' qubits are taken in reverse order here, because the transform used leaves them that way.

So adding two registers costs only CNOT and Toffoli gates, with no rotation to approximate, which is the difference from
`fourier`, where an add needs the other operand as plain bits and costs rotations. What it moves elsewhere is the
conversions: an add needs its operands in the Fourier basis, so a register loaded as plain bits is converted first, and
anything that reads bits (a comparison, a branch, the returned value) converts back.

Multiplication adds the multiplicand into the product once per bit of the multiplier, shifted, and that bit controls the
gates, so the multiplier must be plain bits. Adding a constant is the same rotations as in `fourier`.

The lowering is the Fourier one with its builder subclassed: the transform, the tracking of which basis each register is
in, and everything that works on plain bits are shared.
"""
from typing import Optional, Sequence

from qiskit import QuantumCircuit

from compiler.compiler import CIRCUIT, ISA, compile_pass
from compiler.parser.instructions import IsaProgram
from compiler.passes.qasm import CircuitError, Q, _constant, _same
from compiler.system import System
from experiments.encodings.passes.fourier import _FourierBuilder, lower


class _DualBuilder(_FourierBuilder):
    def is_arithmetic(self, opcode: str, srcs) -> bool:
        """A copy of a register that is already in the Fourier basis stays there, as an add into a zero register; one of plain bits is a plain copy."""
        if opcode == "copy":
            return _constant(srcs[0]) is None and self.slot([bit.index for bit in srcs[0]]) in self.basis
        return super().is_arithmetic(opcode, srcs)

    def prepare(self, opcode: str, srcs) -> None:
        """An add or subtract needs its register operands in the Fourier basis. A multiply needs the multiplier as plain bits."""
        for k, source in enumerate(srcs):
            if _constant(source) is None:
                qubits = [bit.index for bit in source]
                if opcode == "mul" and k == 1 and _constant(srcs[0]) is None:
                    self.to_plain(qubits)  # the multiplier controls the gates
                else:
                    self.to_fourier(qubits)

    def dual_add(self, dst: Sequence[int], src: Sequence[int], sign: int, shift: int = 0) -> None:
        """
        dst += sign * src * 2^shift, with both registers in the Fourier basis: the binary subtractor (for +) or adder (for -)
        with `src` as the accumulator and `dst` as the addend, bit i of each register being the qubit that carries bit i
        once the transform's reversal is undone.
        """
        n = len(dst)
        accumulator, addend = list(reversed(src))[shift:], list(reversed(dst))[:n - shift]
        if sign > 0:
            with self.undone():
                self.adder(accumulator, addend, self.scratch)
        else:
            self.adder(accumulator, addend, self.scratch)

    def accumulate(self, acc: Sequence[int], source, sign: int) -> None:
        """acc += sign * source, for a constant (rotations) or a register in the Fourier basis (the swapped binary adder)."""
        constant = _constant(source)
        if constant is not None:
            self.rotate(acc, sign * constant)
        else:
            self.dual_add(acc, [bit.index for bit in source], sign)

    def arithmetic(self, opcode: str, dst: Sequence[int], srcs) -> None:
        if opcode == "copy":
            return self.accumulate(dst, srcs[0], 1)
        if opcode != "mul":
            return super().arithmetic(opcode, dst, srcs)
        a, b = srcs
        if _constant(a) is not None:
            a, b = b, a
        constant = _constant(b)
        if constant is not None:  # the register is added in once for each set bit of the constant
            for j in range(self.w):
                if (constant >> j) & 1:
                    self.dual_add(dst, [bit.index for bit in a], 1, shift=j)
        else:  # a is added in, shifted, for each bit of the multiplier b
            for j, multiplier in enumerate(b):
                with self.controlled_on([(Q(multiplier.index), True)]):
                    self.dual_add(dst, [bit.index for bit in a], 1, shift=j)

    def update(self, opcode: str, dst: Sequence[int], src) -> None:
        constant = _constant(src)
        if constant is None:
            if _same([Q(q) for q in dst], src):
                raise CircuitError("a register cannot be updated with itself: the result cannot be undone")
            self.to_fourier([bit.index for bit in src])
        self.to_fourier(dst)
        self.accumulate(dst, src, 1 if opcode == "add" else -1)


@compile_pass(
    ISA, CIRCUIT,
    title="Lower to a Fourier-encoded circuit (adder with roles swapped)",
    summary=(
        "Integers held as phases; adding two registers runs the ordinary binary subtractor with its roles swapped, "
        "which needs only CNOT and Toffoli gates but needs both registers in the Fourier basis."
    ),
)
def to_circuit_fourier_dual(program: IsaProgram, system: Optional[System] = None) -> QuantumCircuit:
    return lower(program, system, _DualBuilder)

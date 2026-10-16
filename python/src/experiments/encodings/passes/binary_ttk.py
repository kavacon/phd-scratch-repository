"""
Binary encoding with the Takahashi, Tani and Kunihiro adder (Quantum Information and Computation 10, 2010): the same
registers and the same circuits as the shared binary lowering, except that adding two registers needs no carry qubit.

The adder is in place (`b += a` with `a` restored) and uses only CNOT and Toffoli gates, 5n-6 and 2n-2 of them for n
bits. The paper's version also writes the carry out of the top bit to a target qubit, which modular arithmetic does
not need, so those two gates are left out. Comparison still borrows a carry qubit from scratch, so a program without
a comparison never touches the scratch register.

This is a local copy of the shared lowering's plumbing (control regions, inverses, scratch, measurement) with the
builder's adder replaced; the shared pass is untouched.
"""
from typing import List, Optional, Sequence, Tuple

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from compiler.compiler import CIRCUIT, ISA, compile_pass
from compiler.parser.instructions import (
    Alloc, Ctrl, EndCtrl, Free, ICtrl, IOp, IsaProgram, Op, Reg, Scratch, Update,
)
from compiler.passes.qasm import CircuitError, Q, _bits, _Builder
from compiler.system import System
from experiments.encodings.readout import Readout


class _TtkBuilder(_Builder):
    def adder(self, acc: Sequence[int], add: Sequence[int], carry: int) -> None:
        """acc += add (modulo 2^n), `add` restored, no ancilla. `carry` is accepted and unused."""
        n = len(acc)
        a, b = add, acc
        cx = lambda control, target: self.x(target, [(control, True)])  # noqa: E731
        ccx = lambda c1, c2, target: self.x(target, [(c1, True), (c2, True)])  # noqa: E731
        if n == 0:
            return
        if n == 1:
            cx(a[0], b[0])
            return
        for i in range(1, n):
            cx(a[i], b[i])
        for i in range(n - 2, 0, -1):
            cx(a[i], a[i + 1])
        for i in range(n - 1):
            ccx(a[i], b[i], a[i + 1])  # the carries, built up in the addend's own qubits
        for i in range(n - 1, 0, -1):
            cx(a[i], b[i])
            ccx(a[i - 1], b[i - 1], a[i])  # and taken apart again, leaving the addend as it was
        for i in range(1, n - 1):
            cx(a[i], a[i + 1])
        for i in range(n):
            cx(a[i], b[i])


@compile_pass(
    ISA, CIRCUIT,
    title="Lower to a binary circuit (no-ancilla adder)",
    summary="Binary registers with the Takahashi, Tani and Kunihiro adder, which needs no carry qubit.",
)
def to_circuit_binary_ttk(program: IsaProgram, system: Optional[System] = None) -> QuantumCircuit:
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

    for instruction in program.body:
        if isinstance(instruction, (Alloc, Free, Scratch)):
            continue
        if isinstance(instruction, EndCtrl):
            regions.pop()
        elif isinstance(instruction, (Op, IOp)):
            dst = [bit.index for bit in _bits(Reg(instruction.dst), width, needed)]
            sources = [_bits(source, width, needed) for source in instruction.sources]
            target = circuit if isinstance(instruction, Op) else QuantumCircuit(*circuit.qregs)
            _TtkBuilder(target, width, scratch, regions).op(instruction.opcode, dst, sources)
            if isinstance(instruction, IOp):
                circuit.compose(target.inverse(), inplace=True)
        elif isinstance(instruction, Update):
            dst = [bit.index for bit in _bits(Reg(instruction.dst), width, needed)]
            _TtkBuilder(circuit, width, scratch, regions).update(instruction.opcode, dst, _bits(instruction.source, width, needed))
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
    return Readout(shots=1, decode="signed-bits").attach(circuit)

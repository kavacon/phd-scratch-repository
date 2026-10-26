"""
Two compilers' circuits for the same task, side by side: each is built, counted the same way, drawn, and run on Qiskit Aer on
request. An experiment builds the two circuits (ours and Qrisp's) and gives them to `Built`; `demo.comparison_blocks` shows it.
"""
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from checking import circuits, qiskit_tools
from checking.qrisp_tools import QrispCircuit
from compiler.passes.qasm import openqasm_text
from demo.drawing import diagram, qiskit_diagram

SIDES = ("ours", "qrisp")
VARIANTS = ("built", "optimised")


@dataclass(frozen=True)
class Side:
    """One compiler's circuit for a program, as built and as optimised, with what is needed to draw and run it."""
    built: Any
    optimised: Any
    result_bits: int
    exponent: int  # a Qrisp QuantumFloat holds mantissa * 2 ** exponent
    text: Dict[str, str]  # the OpenQASM of each variant
    drawings: Dict[str, Dict[str, Any]]
    names: Tuple[str, ...]  # the register or variable each qubit of the built circuit belongs to

    def variant(self, name: str):
        return self.built if name == "built" else self.optimised


@dataclass(frozen=True)
class Built:
    expected: Any  # the value the program returns, or a dictionary of each value's probability when it returns a distribution
    sources: Dict[str, str]
    sides: Dict[str, Side]


def text_of(circuit) -> str:
    try:
        return openqasm_text(circuit)
    except Exception as error:  # a circuit with gates OpenQASM cannot name
        return f"// this circuit cannot be written as OpenQASM: {error}"


def ours_side(generated, optimised, result_bits: int) -> Side:
    names = tuple(generated.find_bit(q).registers[0][0].name for q in generated.qubits)
    return Side(generated, optimised, result_bits, 0,
                {"built": text_of(generated), "optimised": text_of(optimised)},
                {"built": diagram(generated), "optimised": diagram(optimised)}, names)


def qrisp_side(qrisp: QrispCircuit) -> Side:
    optimised = qiskit_tools.optimise(qrisp.circuit)
    return Side(qrisp.circuit, optimised, len(qrisp.result), qrisp.exponent,
                {"built": text_of(qrisp.circuit), "optimised": text_of(optimised)},
                {"built": qiskit_diagram(qrisp.circuit, qrisp.labels, qrisp.groups),
                 "optimised": qiskit_diagram(optimised, qrisp.labels, qrisp.groups)},
                tuple(label.split(".")[0] for label in qrisp.labels))


def metrics(built: Built, side: str, variant: str) -> Dict[str, Any]:
    """Gate and wire counts of one of the four circuits, counted the same way."""
    return qiskit_tools.measure(built.sides[side].variant(variant))


def run_circuit(built: Built, side: str, variant: str, shots: int) -> Dict[str, Any]:
    """Run one of the four circuits on Qiskit Aer and report the measurement against what the program should return."""
    if side not in SIDES or variant not in VARIANTS:
        return {"error": "unknown circuit"}
    if not qiskit_tools.aer_available():
        return {"error": "Running a circuit needs Qiskit Aer: pip install qiskit-aer"}
    shots = min(max(shots, 1), qiskit_tools.MAX_SHOTS)
    entry = built.sides[side]
    outcome = qiskit_tools.execute(entry.variant(variant), entry.result_bits, shots)
    scale = 2 ** entry.exponent
    if isinstance(built.expected, dict):
        return {"side": side, "variant": variant, **circuits.distribution_report(outcome, shots=shots, expected=built.expected, scale=scale)}
    return {"side": side, "variant": variant, **circuits.report(outcome, shots=shots, expected=built.expected, scale=scale)}

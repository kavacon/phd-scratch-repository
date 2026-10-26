"""
Run a lowered program and read the measured register back as an integer.

The module goes through MQT's compiler (QC in generic syntax in, OpenQASM 3 out) and is then run on Qiskit Aer by the
repo's own checker, so the simulator is independent of MQT and the exporter is exercised too.
"""
from typing import Any, Dict

from mqt.core.mlir import QCProgram
from xdsl.dialects.builtin import ModuleOp

from checking import qiskit_tools
from checking.circuits import to_signed
from experiment_existing_ir.ir import show


def mqt(module: ModuleOp) -> QCProgram:
    return QCProgram.from_mlir_str(show(module, generic=True))


def openqasm(module: ModuleOp) -> str:
    """
    The program as OpenQASM 3 text from MQT's exporter. It marks the result register `output`, which Qiskit's importer
    does not accept, so that keyword is dropped; nothing else needs changing.
    """
    return mqt(module).to_openqasm3().source.replace("output bit", "bit")


def execute(module: ModuleOp) -> Dict[str, Any]:
    """
    The value of the result register (two's complement, least significant bit first; the simulator's key reads the
    highest bit first) and the qubits left set outside it. Every qubit is in a basis state, so a few shots must agree.
    """
    width = next(op for op in module.ops if op.name == "func.func" and op.sym_name.data == "main"
                 ).function_type.outputs.data[0].width.data
    outcome = qiskit_tools.execute(qiskit_tools.read(openqasm(module)), width)
    if not outcome["deterministic"]:
        raise ValueError(f"the result is not a single value: {outcome['counts']}")
    return {"value": to_signed(outcome["value"], width), "leaked": outcome["leaked"]}


def run(module: ModuleOp) -> int:
    return execute(module)["value"]

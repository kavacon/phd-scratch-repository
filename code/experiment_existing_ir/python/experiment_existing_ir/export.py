"""
The QC program out of the tool as three interchange formats, and each exported text run on a simulator that did not produce it.

MQT's exporters turn the QC program into OpenQASM 3 and into QIR (the base profile, with no branching, and the adaptive
profile). MQT's own device for DDSIM, a decision-diagram simulator, accepts all three, so each exported text is submitted to it
as it is written out, which checks the exporter and not the QC program. OpenQASM 3 also runs on Qiskit's Aer. Nothing here
changes the program; a format that cannot hold it is reported by the error the exporter gives.

    . ./env.sh && .venv/bin/python -m experiment_existing_ir.interchange
"""
from typing import Callable, Dict, Tuple

from mqt.core import qdmi
from mqt.core.mlir import QCProgram, QIRProfile

from checking.circuits import to_signed
from experiment_existing_ir import simulate

DEVICE = "mqt.ddsim.default"
FORMATS: Tuple[str, ...] = ("openqasm3", "qir-base", "qir-adaptive")
_PROGRAM_FORMAT = {"openqasm3": qdmi.ProgramFormat.QASM3, "qir-base": qdmi.ProgramFormat.QIR_BASE_STRING,
                   "qir-adaptive": qdmi.ProgramFormat.QIR_ADAPTIVE_STRING}


def export(qc: str, format: str) -> str:
    """The QC program as text in the format."""
    program = QCProgram.from_mlir_str(qc)
    if format == "openqasm3":
        return simulate.openqasm(qc)
    profile = {"qir-base": QIRProfile.BASE, "qir-adaptive": QIRProfile.ADAPTIVE}[format]
    return program.to_qir(profile, copy=True).llvm_ir


def on_ddsim(text: str, format: str, shots: int = 2000) -> Dict[int, int]:
    """How often each value of the result register came out when the exported text is run on DDSIM."""
    job = qdmi.open_device(DEVICE).submit_job(text, _PROGRAM_FORMAT[format], shots)
    job.wait()
    counts = job.get_counts()
    width = len(next(iter(counts)))
    return {to_signed(value, width): count for value, count in sorted((int(key, 2), n) for key, n in counts.items())}


def on_aer(text: str, shots: int = 2000) -> Dict[int, int]:
    """The same for OpenQASM 3 run on Qiskit's Aer, through the importer."""
    from checking import qiskit_tools
    width = int(text.split("bit[")[1].split("]")[0])
    counts = qiskit_tools.execute(qiskit_tools.read(text.replace("output bit", "bit")), width, shots=shots)["counts"]
    return {to_signed(value, width): count for value, count in sorted(counts.items())}


BACKENDS: Dict[str, Callable[[str, str, int], Dict[int, int]]] = {
    "ddsim": lambda text, format, shots: on_ddsim(text, format, shots),
    "aer": lambda text, format, shots: on_aer(text, shots),
}

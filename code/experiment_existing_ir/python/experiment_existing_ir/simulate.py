"""
Run a program through the C++ tool and read the measured register back as integers.

`existing-ir-opt` (built from `cpp/`) is `mlir-opt` with MQT's dialects and passes, our `dist` dialect and the
`lower-to-qc` pass. Its output is QC text, which goes through MQT's compiler (QC in, OpenQASM 3 out) and is then run on
Qiskit Aer by the repo's own checker, so the simulator is independent of MQT and the exporter is exercised too.
"""
import subprocess
from pathlib import Path
from typing import Dict, Sequence

from mqt.core.mlir import QCProgram

from checking import qiskit_tools
from checking.circuits import to_signed

TOOL = Path(__file__).resolve().parents[3] / "build/experiment_existing_ir/existing-ir-opt"


def built() -> bool:
    return TOOL.exists()


def optimise(program: str, *passes: str) -> str:
    """The program after the given passes (`--canonicalize`, `--lower-to-qc`, ...), as text."""
    return subprocess.run([TOOL, *passes], input=program, capture_output=True, text=True, check=True).stdout


def openqasm(qc: str) -> str:
    """
    The QC program as OpenQASM 3 text from MQT's exporter. It marks the result register `output`, which Qiskit's importer
    does not accept, so that keyword is dropped; nothing else needs changing.
    """
    return QCProgram.from_mlir_str(qc).to_openqasm3().source.replace("output bit", "bit")


def execute(qc: str, shots: int = 2000) -> Dict[int, int]:
    """
    How often each value of the result register came out (two's complement, least significant bit first; the
    simulator's key reads the highest bit first). A definite program gives one value; a distribution gives several.
    """
    qasm = openqasm(qc)
    width = int(qasm.split("bit[")[1].split("]")[0])
    counts = qiskit_tools.execute(qiskit_tools.read(qasm), width, shots=shots)["counts"]
    return {to_signed(value, width): count for value, count in sorted(counts.items())}


def run(program: str, passes: Sequence[str] = ("--resolve-instances", "--canonicalize", "--in-place-updates",
                                                "--lower-to-qc"), shots: int = 2000) -> Dict[int, int]:
    return execute(optimise(program, *passes), shots)

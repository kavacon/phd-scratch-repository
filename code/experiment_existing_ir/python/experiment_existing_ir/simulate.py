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
from experiment_existing_ir.translate import MAX_DEPTH

TOOL = Path(__file__).resolve().parents[3] / "build/experiment_existing_ir/existing-ir-opt"


# Upstream's pass for `affine.for`, nested in a function because it works on one. A factor of -1 unrolls a loop fully, and
# each repetition reaches one level further out, since it only takes the innermost loops (the translation refuses a nest
# deeper than `translate.MAX_DEPTH`). The `unroll-full-threshold` option, tried first, skips loops of fewer than four trips.
UNROLL = "func.func(affine-loop-unroll{unroll-factor=-1 unroll-num-reps=%d})" % MAX_DEPTH


# Turns the slots of the translation into SSA values, so every pass after it sees values. A loop has to be unrolled first,
# since the pass does not promote across an `affine.for`.
PROMOTE = "func.func(mem2reg)"
# What the translation needs before anything else: loops unrolled, then calls inlined and their functions dropped, then
# slots promoted. The inliner refuses a call inside an `affine.for`, so the loops go first.
PREPARE = (UNROLL, "inline", "symbol-dce", PROMOTE)


def built() -> bool:
    return TOOL.exists()


# The tool's own passes that are anchored on a function. Flags nest them by themselves; in a pipeline they are nested here.
FUNCTION_PASSES = {"resolve-instances", "tie-instances", "in-place-updates", "dist-status", "resolve-refs", "flatten-dist-if",
                   "check-resolved"}


def command(*passes: str) -> list:
    """
    The tool's command line for the given passes (`--canonicalize`, `--lower-to-qc`, `UNROLL`, ...) run in that order. The
    tool takes either flags or one pipeline, and a pass nested in a function needs the pipeline, so the passes become one.
    Flags that are not passes (`--mlir-print-ir-after-all`) are kept as they are.
    """
    options = [p for p in passes if p.startswith("--mlir")]
    names = [p.removeprefix("--") for p in passes if not p.startswith("--mlir")]
    names = [f"func.func({name})" if name in FUNCTION_PASSES else name for name in names]
    return [TOOL, *options, *([f"--pass-pipeline=builtin.module({','.join(names)})"] if names else [])]


def optimise(program: str, *passes: str) -> str:
    """The program after the given passes, as text."""
    return subprocess.run(command(*passes), input=program, capture_output=True, text=True, check=True).stdout


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


# The default pipeline. Resolving the references: `resolve-refs` repeats `canonicalize` (which folds the conditions that are definite, forwards the
# value an object holds to the markers and the reads that depend on it, and turns arithmetic on lifted integers back into
# integers), `mem2reg` (which promotes the objects nothing escapes any more) and `flatten-dist-if` (which takes the
# conditionals that stay on a superposition) until nothing changes. `check-resolved` refuses what is still open.
RESOLVE = ("--resolve-refs", "--check-resolved")
DEFAULT = (*PREPARE, *RESOLVE, "--resolve-instances", "--canonicalize", "--in-place-updates", "--lower-to-qc")


def run(program: str, passes: Sequence[str] = DEFAULT, shots: int = 2000) -> Dict[int, int]:
    return execute(optimise(program, *passes), shots)

"""
The compiler as a list of stages, and the means to run it.

A program goes through `existing-ir-opt` (built from `cpp/`: `mlir-opt` with MQT's dialects and passes, our `dist` dialect and the
`lower-to-qc` pass) as a list of its passes, which ends in QC text. A final stage may be an `Export`, MQT's exporter for one
interchange format, so that an output format is a pass like any other: `(*DEFAULT, Export("qir-base"))` compiles a program to
QIR. The artifact is text, or bytes for Jeff. Running it is separate: the simulators here read an artifact as it was written out
and give back the measured register as integers.
"""

import subprocess
from pathlib import Path
from typing import Callable, Dict, NamedTuple, Optional, Sequence, Tuple, Union

from compiler.system import System
from mqt.core import qdmi
from mqt.core.mlir import JeffProgram, OutputFormat, QCProgram, QIRProfile, compile_program, sample

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
FUNCTION_PASSES = {"resolve-instances", "tie-instances", "in-place-updates", "dist-status", "resolve-dist", "qubit-liveness"}


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


# The default pipeline. `resolve-dist` runs one dataflow analysis (which branch is live, what each value is, what each cell holds,
# which cell each reference names) and rewrites once from what it found; it refuses what the analysis could not decide.
# Stock `canonicalize` then folds the constant selects and removes what is dead.
RESOLVE = ("--resolve-dist", "--sccp", "--canonicalize")
# MQT's own pass that turns an inverse of several gates (or of a call) into the inverted gates in reverse order, and `canonicalize`
# then folds each single-gate inverse (a self-adjoint gate is itself). The lowering writes a subtraction as the adder inside `qc.inv`,
# which the exporters (DDSIM's OpenQASM reader, Jeff) do not take; after this the circuit has no modifier on a call.
UNROLL_MODIFIERS = ("--unroll-modifiers", "--canonicalize")
BEFORE_LOWERING = (*PREPARE, *RESOLVE, "--resolve-instances", "--canonicalize", "--cse", "--in-place-updates")
DEFAULT = (*BEFORE_LOWERING, "--dist-status", "--plan-registers", "--classify-release", "--lower-to-qc", *UNROLL_MODIFIERS, "--assign-qubits")


class Export(NamedTuple):
    """A stage that ends a pipeline: the QC program written out in an interchange format (`FORMATS`)."""
    format: str


FORMATS: Tuple[str, ...] = ("openqasm3", "qir-base", "qir-adaptive", "jeff")


def with_system(program: str, system: System) -> str:
    """The program inside a module that describes the system with MLIR's own target information (the `dlti` dialect): the
    qubits it has. `lower-to-qc` reads it and refuses a program that needs more at once. (The system's register width belongs to
    the older fixed-width compiler; here widths come from ranges, so only the qubits are described.)"""
    spec = f'#dlti.target_system_spec<"quantum" = #dlti.target_device_spec<"qubits" = {system.qubits} : i32>>'
    return f"module attributes {{dlti.target_system_spec = {spec}}} {{\n{program}\n}}"


def build(program: str, *stages: Union[str, Export], system: Optional[System] = None) -> Union[str, bytes]:
    """The program after its stages: QC text, or the exported artifact when the last stage is an `Export`. Given a system, the
    tool refuses a program that needs more qubits than the system has (`subprocess.CalledProcessError`, whose `stderr` says by how
    much and which registers)."""
    *passes, last = stages if stages and isinstance(stages[-1], Export) else (*stages, None)
    qc = optimise(program if system is None else with_system(program, system), *passes)
    return qc if last is None else export(qc, last.format)


def export(qc: str, format: str) -> Union[str, bytes]:
    """MQT's exporter: the QC program in the format. OpenQASM 3 marks the result register `output`, which Qiskit's importer does
    not accept, so that keyword is dropped; nothing else needs changing."""
    program = QCProgram.from_mlir_str(qc)
    if format == "jeff":
        return compile_program(program, output=OutputFormat.JEFF).to_bytes()
    if format == "openqasm3":
        return program.to_openqasm3().source.replace("output bit", "bit")
    profile = {"qir-base": QIRProfile.BASE, "qir-adaptive": QIRProfile.ADAPTIVE}[format]
    return program.to_qir(profile, copy=True).llvm_ir


# ---- running an artifact -----------------------------------------------------------------------------------------------------
# The simulators here read an artifact as written out, so the exporter is exercised too: Qiskit's Aer reads OpenQASM 3 through its
# own importer (independent of MQT); MQT's DDSIM device reads OpenQASM 3 and both QIR profiles as text; Jeff only MQT reads, so it is
# read back and sampled by MQT's decision-diagram simulator.
DEVICE = "mqt.ddsim.default"
_PROGRAM_FORMAT = {"openqasm3": qdmi.ProgramFormat.QASM3, "qir-base": qdmi.ProgramFormat.QIR_BASE_STRING,
                   "qir-adaptive": qdmi.ProgramFormat.QIR_ADAPTIVE_STRING}


def _values(counts: Dict[str, int]) -> Dict[int, int]:
    """Counts keyed by the measured bit string (highest bit first) as two's complement integers."""
    width = len(next(iter(counts)))
    return {to_signed(value, width): count for value, count in sorted((int(key, 2), n) for key, n in counts.items())}


def on_aer(text: str, format: str, shots: int) -> Dict[int, int]:
    width = int(text.split("bit[")[1].split("]")[0])
    counts = qiskit_tools.execute(qiskit_tools.read(text), width, shots=shots)["counts"]
    return {to_signed(value, width): count for value, count in sorted(counts.items())}


def on_ddsim(text: str, format: str, shots: int) -> Dict[int, int]:
    job = qdmi.open_device(DEVICE).submit_job(text, _PROGRAM_FORMAT[format], shots)
    job.wait()
    return _values(job.get_counts())


def on_mqt(data: bytes, format: str, shots: int) -> Dict[int, int]:
    return _values(sample(JeffProgram.from_bytes(data), shots=shots))


BACKENDS: Dict[str, Callable[[Union[str, bytes], str, int], Dict[int, int]]] = {"aer": on_aer, "ddsim": on_ddsim, "mqt": on_mqt}


def execute(artifact: Union[str, bytes], format: str = "openqasm3", backend: str = "aer", shots: int = 2000) -> Dict[int, int]:
    """How often each value of the result register came out when the artifact is run: a definite program gives one value, a
    distribution gives several."""
    return BACKENDS[backend](artifact, format, shots)


def run(program: str, passes: Sequence[str] = DEFAULT, shots: int = 2000) -> Dict[int, int]:
    """The program compiled to OpenQASM 3 and run on Aer."""
    return execute(build(program, *passes, Export("openqasm3")), shots=shots)

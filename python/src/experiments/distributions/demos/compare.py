"""
Qrisp comparison: our circuits next to Qrisp's for the same task at 8-bit registers, built, counted and optionally run the same way.

    PYTHONPATH=src .venv/bin/python -m experiments.distributions.demos.compare

It needs the optional dependencies: pip install '.[circuits,compare]'
"""
from typing import Any, Dict, List, Sequence

from checking import qiskit_tools
from demo import app, comparison_blocks
from demo.blocks import measurement
from demo.model import Action, Bar, Case, Cell, Code, Columns, Note, Number, Result, Row, Stage, Stats
from experiments.distributions import qrisp_comparison as comparison
from experiments.distributions.programs import PROGRAMS

EXPLANATION = (
    "Ours: the whole pipeline of the experiment, from the source with its ranges to a circuit: constants folded, steps combined, "
    "values released as soon as they can be undone, conditionals on dedicated bits, then gates (Hadamards or a state preparation for a "
    "range, a Cuccaro adder, shift-and-add multiplication).",
    "Qrisp: the same task written by hand as ordinary Qrisp, using only what Qrisp provides: QuantumFloat registers, `h` or "
    "`init_state` for a distribution, its own +, - , * and comparison operators, `with condition:` for an if, and "
    "@auto_uncompute for the temporaries. Constants are Python numbers. Nothing is wrapped or annotated, so the circuit is what "
    "Qrisp's compiler makes of the code, with its own adders, qubit allocation and ancillae.",
    "So the gate constructions are not the same. This compares the two generation processes as they come, not matched adders.",
    "Registers are 8 bits. Our arithmetic wraps modulo 256 and Qrisp sizes the results of out-of-place arithmetic itself, so a "
    "product has more qubits than a register. Every value here is small enough that nothing wraps, so both return the same "
    "distribution, and the difference is in the qubit and gate counts. Where Qrisp refuses to multiply by 2 and then add, the "
    "Qrisp text writes `b + b`. The measured value is the mantissa times two to the exponent.",
    "Running a circuit compares the fractions of shots that returned each value with the probability the instruction machine gives it. "
    "Both leave the registers that hold the distribution set, because the result is entangled with them and cannot be undone, "
    "so some qubits are set at the end in some shots; that is expected.",
    "Same measuring: every circuit is read into Qiskit and counted the same way, as built (a multi-controlled X, or a Qrisp "
    "QFT, counts as one gate) and after Qiskit's optimiser into u and cx. The as-built counts are not comparable gate for gate: "
    "ours holds additions and comparisons as a few multi-controlled X gates, which Qiskit later breaks into many, while "
    "Qrisp's arithmetic is phase based (a Fourier transform, then rotations) and its controlled parts are already written out "
    "as Toffoli ladders on workspace qubits. The optimised counts are the like-for-like ones.",
    "Releases: in Qrisp only the temporaries inside the function are uncomputed. The registers that hold the distributions are "
    "made outside it, so Qrisp never releases them, where ours releases them whenever it can be done by undoing steps. The "
    "state check below shows what each circuit really leaves behind.",
)

STATE_EXPLANATION = (
    "The exact probability of each state a circuit ends in (the circuit as built, with the measurement left out), by register. "
    "It lists the registers that are not zero at the end, and for each whether the result is entangled with it: the "
    "probabilities of the two together are not the product of theirs, which an unentangled state cannot give. "
    "A held register that is not entangled with the result could have been released by undoing it, and one the result determines "
    "(each result value goes with one value of the register) by computing it back from the result. Both are missed releases. One "
    "that is entangled and not determined is garbage a unitary program cannot avoid. Phases are not shown.",
    "For ours the states are also compared with those of the instruction machine for the allocated program, register by register.",
)

MAX_STATES = 12


def _state_blocks(report: Dict[str, Any]) -> List[Any]:
    result, held, scale = report["registers"][0], report["held"], report["scale"]
    blocks: List[Any] = []
    reference = report["reference"]
    if reference is not None:
        blocks.append(Note("The states match the instruction machine's for the allocated program." if reference["matches"]
                           else "The states differ from the instruction machine's for the allocated program.",
                           "good" if reference["matches"] else "bad"))
    if not held:
        blocks.append(Note(f"Everything except the result is zero again: the other {report['qubits'] - report['result_qubits']} "
                           "qubits it used are released.", "good"))
    for name, info in held.items():
        what = f"{name} ({info['qubits']} qubits) still holds a value"
        if not info["entangled"]:
            blocks.append(Note(f"{what} that is not entangled with the result: it could have been released by undoing it.", "bad"))
        elif info["determined"]:
            blocks.append(Note(f"{what}, entangled with the result. The result determines it, so it could be released by computing "
                               "it back from the result.", "bad"))
        else:
            blocks.append(Note(f"{what}, entangled with the result, which does not determine it. It is garbage a unitary "
                               "program cannot release without changing the result.", "muted"))
    names = list(held)
    columns = [report["registers"].index(name) for name in names]
    rows = []
    for *values, p in report["states"][:MAX_STATES]:
        label = " · ".join([f"{result} = {values[0] * scale}"] + [f"{name} = {values[k]}" for name, k in zip(names, columns)])
        rows.append(Row(label, (Cell(f"{p:.1%}", bars=(Bar(p, "series-1"),)),)))
    blocks.append(Stats(("state", "probability"), tuple(rows)))
    if len(report["states"]) > MAX_STATES:
        blocks.append(Note(f"{len(report['states']) - MAX_STATES} more states are not shown.", "muted"))
    return blocks



def _case(inputs: Dict[str, Any]) -> str:
    return inputs["case"] or PROGRAMS[0]["id"]


def run(inputs: Dict[str, Any], enabled: List[str]) -> Result:
    program, width = _case(inputs), inputs["width"]
    try:
        built = comparison.build(program, width)
    except Exception as error:  # Qrisp or Qiskit missing, or a program Qrisp cannot build
        return Result((Stage("error", "Could not build the circuits", blocks=(Note(f"{type(error).__name__}: {error}", "bad"),), failed=True),))
    return Result((
        Stage("programs", "The two programs",
              "The same task, built two ways: our compiler turns the program on the left into a circuit, and Qrisp's own compiler "
              "turns the Qrisp program on the right into one. Both are measured the same way below.",
              (Columns((
                  (Code.of(built.sources["ours"], "Ours · source language", "ranges are distributions"),),
                  (Code.of(built.sources["qrisp"], "Qrisp · Python", "@auto_uncompute"),),
              )), Note("What is and is not equivalent", "muted", EXPLANATION))),
        Stage("statistics", "Statistics", "", tuple(comparison_blocks.statistics(built))),
        Stage("circuits", "Circuits", "", (comparison_blocks.circuits(built),)),
        Stage("state", "Final state and releases",
              "What each circuit leaves behind: its final state by register, and whether what it did not release could have been.",
              (Note("How this is read", "muted", STATE_EXPLANATION),
               Columns(tuple((Action(f"Show the final state: {title}", "state", {"side": side}, (), "Qiskit Aer"),)
                             for side, title in (("ours", "ours"), ("qrisp", "Qrisp")))))),
    ))


def simulate(inputs: Dict[str, Any], enabled: List[str], payload: Dict[str, Any]) -> Sequence[Any]:
    try:
        shots = int(payload.get("shots", 100))
    except (TypeError, ValueError):
        shots = 100
    report = comparison.run_circuit(_case(inputs), inputs["width"], str(payload.get("side", "")), str(payload.get("variant", "")), shots)
    return measurement(report)


def state(inputs: Dict[str, Any], enabled: List[str], payload: Dict[str, Any]) -> Sequence[Any]:
    side = str(payload.get("side", ""))
    if side not in comparison.SIDES:
        return [Note("unknown circuit", "bad")]
    if not qiskit_tools.aer_available():
        return [Note("Reading the final state needs Qiskit Aer: pip install qiskit-aer", "bad")]
    try:
        return _state_blocks(comparison.final_state(_case(inputs), inputs["width"], side))
    except Exception as error:
        return [Note(f"{type(error).__name__}: {error}", "bad")]


DEMO = app.Demo(
    title="Compiler comparison: ours and Qrisp",
    description="The same task, returning a distribution, built by our compiler and by Qrisp's, with the circuits and counts side by side.",
    run=run,
    inputs=(Number("width", "Register width (qubits)", comparison.DEFAULT_WIDTH, 2, comparison.MAX_WIDTH,
                   "Both circuits use registers this wide, unsigned, modulo 2 to this width."),),
    cases=tuple(Case(p["id"], p["name"], p["description"]) for p in PROGRAMS),
    actions={"simulate": simulate, "state": state},
    workflow=False,
    cases_label="Programs",
    port=8005,
)


def make_server(port: int):
    return app.make_server(DEMO, port)


def main() -> None:
    app.serve(DEMO)


if __name__ == "__main__":
    main()

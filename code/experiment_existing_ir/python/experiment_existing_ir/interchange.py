"""
Each program of the semantics oracle through every export format and simulator, against the reference outcome law.

For a program and a format this reports whether the exporter accepted it (and the reason when it did not), whether the exported
text, run on a simulator that did not produce it, gives the outcome the reference works out from the source, and the size of
the exported text. A format that cannot hold a program is reported as refused, never as a failure; a mismatch is a wrong answer.

    . ./env.sh && .venv/bin/python -m experiment_existing_ir.interchange
"""
from typing import Dict, List, NamedTuple

from compiler.parser.parser import parse
from experiment_existing_ir import export, simulate
from experiment_existing_ir.reference import ReferenceError_, law
from experiment_existing_ir.semantics import PROGRAMS, TOLERANCE, distance
from experiment_existing_ir.translate import TranslationError, translate

SHOTS = 2000
# (format, simulator) pairs: OpenQASM 3 runs on both; QIR only on DDSIM, the one simulator here that reads it.
RUNS = (("openqasm3", "aer"), ("openqasm3", "ddsim"), ("qir-base", "ddsim"), ("qir-adaptive", "ddsim"))


class Outcome(NamedTuple):
    status: str  # "ok", "WRONG" or "refused"
    detail: str = ""
    size: int = 0  # characters of exported text


def check(qc: str, expected: Dict[int, float], format: str, backend: str) -> Outcome:
    try:
        text = export.export(qc, format)
    except Exception as error:  # the exporter does not take the program
        return Outcome("refused", str(error).strip().splitlines()[0][:100] if str(error).strip() else type(error).__name__)
    try:
        counts = export.BACKENDS[backend](text, format, SHOTS)
    except Exception as error:
        return Outcome("refused", str(error).strip().splitlines()[0][:100] if str(error).strip() else type(error).__name__, len(text))
    gap = distance(counts, expected)
    return Outcome("ok" if gap < TOLERANCE else "WRONG", f"distance {gap:.2f}", len(text))


def main() -> None:
    columns = [f"{format}/{backend}" for format, backend in RUNS]
    print(f"{'program':<56}" + "".join(f"{c:<22}" for c in columns))
    wrong: List[str] = []
    sizes: Dict[str, List[int]] = {c: [] for c in columns}
    reasons: Dict[str, str] = {}
    for name, source in PROGRAMS:
        try:
            expected = law(parse(source, "v2"))
            qc = simulate.optimise(translate(parse(source, "v2")), *simulate.DEFAULT)
        except (ReferenceError_, TranslationError, Exception):
            continue
        outcomes = [check(qc, expected, format, backend) for format, backend in RUNS]
        print(f"{name:<56}" + "".join(f"{o.status:<22}" for o in outcomes))
        for column, outcome in zip(columns, outcomes):
            if outcome.status == "WRONG":
                wrong.append(f"{name} ({column})")
            if outcome.status == "refused":
                reasons.setdefault(column, outcome.detail)
            if outcome.size:
                sizes[column].append(outcome.size)
    print()
    for column in columns:
        if sizes[column]:
            print(f"{column:<26} text size: median {sorted(sizes[column])[len(sizes[column]) // 2]}, max {max(sizes[column])} characters")
        if column in reasons:
            print(f"{'':<26} refused, for example: {reasons[column]}")
    print(f"\nwrong answers: {len(wrong)}" + ("" if not wrong else "  -> " + "; ".join(wrong)))


if __name__ == "__main__":
    main()

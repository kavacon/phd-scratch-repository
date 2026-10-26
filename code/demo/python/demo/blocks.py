"""Blocks several demos build the same way, from data the checking package produces."""
from typing import Any, Dict, List, Optional

from demo.model import Bar, Cell, Note, Row, Stats


def change(now: int, before: Optional[int], unit: str, since: str = "the last stage") -> str:
    """How a number moved since an earlier stage, as a sentence for a card's sub-line; empty when there is nothing to compare."""
    if before is None:
        return ""
    if now == before:
        return f"no change since {since}"
    return f"{'+' if now > before else '−'}{abs(now - before):,} {unit} since {since}"


def measurement(report: Dict[str, Any]) -> List[Any]:
    """A simulator run (checking.circuits.report) as a verdict, any qubits left set, and the measured counts."""
    if "error" in report:
        return [Note(report["error"], "bad")]
    if "distribution" in report:
        return _distribution(report)
    expected = report["expected"]
    if expected is None:
        verdict = Note("No expected value to compare with.")
    elif report["matches"]:
        verdict = Note(f"Every shot returned {expected}, the value the program should return.", "good")
    else:
        verdict = Note(f"Expected {expected}, but the measurement differs.", "bad")
    blocks: List[Any] = [verdict]
    if report["leaked"]:
        blocks.append(Note(
            f"Qubits left set at the end: {len(report['leaked'])} ({', '.join(report['leaked'])}). "
            "They do not change the result, but they were not uncomputed.", "bad"))
    top = max((n for _, n in report["counts"]), default=1)
    rows = tuple(
        Row(str(value), (Cell(str(n)), Cell(bars=(Bar(n / top, "good" if value == expected else "bad"),))))
        for value, n in report["counts"]
    )
    blocks.append(Stats(("measured", "shots", ""), rows))
    return blocks


def _distribution(report: Dict[str, Any]) -> List[Any]:
    """A run of a program that returns a distribution: each value's expected probability beside the fraction measured."""
    close = report["matches"]
    verdict = Note(
        f"The measured fractions are within {report['tolerance']:.2f} of the expected probabilities ({report['distance']:.3f} apart)."
        if close else
        f"The measured fractions are {report['distance']:.3f} from the expected probabilities, more than the {report['tolerance']:.2f} "
        "that chance would give.", "good" if close else "bad")
    blocks: List[Any] = [verdict]
    if report["leaked"]:
        blocks.append(Note(
            f"Qubits that were set at the end in some shots: {len(report['leaked'])}. A program whose result is entangled with its "
            "inputs cannot clear them, so this is expected for those that are.", "muted"))
    rows = tuple(
        Row(str(value), (Cell(f"{expected:.0%} expected"), Cell(f"{measured:.0%} measured",
                                                              bars=(Bar(expected, "series-1"), Bar(measured, "series-2")))))
        for value, expected, measured in report["distribution"])
    blocks.append(Stats(("value", "expected", "measured"), rows))
    return blocks


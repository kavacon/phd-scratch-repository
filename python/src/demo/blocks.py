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

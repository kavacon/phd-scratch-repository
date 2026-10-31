"""
Do the compiled programs mean what the source says?

Each program is given to the reference (`reference.py`, which computes its intended outcome law exactly from the source
alone) and compiled both ways (`comparison.py`: A leaves the folding to MLIR, B folds as it translates). The circuit is run on
Aer, and its sampled outcomes are compared with the reference law by total variation distance. A program a way of compiling
refuses is reported as refused, never as a failure, since refusing is how an unsupported construct is meant to behave; a
mismatch is a wrong answer and is the thing this is looking for.

    . ./env.sh && .venv/bin/python -m experiment_existing_ir.semantics
"""
import random
import sys
from typing import Dict, List, NamedTuple, Optional, Tuple

from compiler.parser.parser import parse
from experiment_existing_ir import simulate
from experiment_existing_ir.reference import ReferenceError_, law
from experiment_existing_ir.translate import TranslationError, translate

SHOTS = 4000
TOLERANCE = 0.07

BUMP = "function bump(p) { p += 1; return p; }\n"

PROGRAMS: Tuple[Tuple[str, str], ...] = (
    # aliases, instances and one roll
    ("alias: update through the second name", "a = 0..4; b = a; b += 1; return a;"),
    ("alias: update through the first name, read the second", "a = 0..4; b = a; a += 1; return b;"),
    ("alias chain", "a = 0..4; b = a; c = b; c += 2; return a;"),
    ("one roll within an expression", "a = 0..4; return a - a;"),
    ("a computed assignment is independent", "a = 0..4; c = a + 1; return c - a;"),
    ("extracting a temporary", "a = 0..4; t = a + 1; c = t + a; return c;"),
    ("two computed assignments are independent", "a = 0..4; c = a + 1; d = a + 1; return c - d;"),
    ("update with another distribution", "a = 0..4; p = 0..2; a += p; return a;"),
    ("an integer is copied", "x = 5; y = x; y += 1; return x;"),
    # conditionals
    ("conditional on a distribution", "a = 0..4; x = 2; if a < 2 { x += 3; } else { x -= 1; } return x;"),
    ("two variables tied to one condition", "a = 0..4; x = 0; y = 10; if a < 2 { x += 1; y -= 1; } return x - y;"),
    ("conditional update of a distribution", "a = 0..4; b = 0..2; if b < 1 { a += 1; } return a;"),
    ("definite conditional, taken", "x = 2; if x < 3 { x += 5; } else { x -= 1; } return x;"),
    # an integer that becomes a distribution
    ("integer updated by a distribution", "a = 0..4; x = 2; x += a; return x - a;"),
    ("integer overwritten by a computed distribution", "a = 0..4; x = 2; x = a + 1; return x - a;"),
    ("taken branch superposes x", "a = 0..4; x = 2; k = 1; if k < 3 { x += a; } return x;"),
    ("dead branch would superpose x", "a = 0..4; x = 2; k = 5; if k < 3 { x += a; } y = x + 1; return y;"),
    ("arm of a distribution conditional superposes x", "a = 0..4; b = 0..2; x = 2; if b < 1 { x += a; } return x;"),
    # loops
    ("integer loop", "x = 1; for i in 0..8 { x += x; } return x;"),
    ("a loop superposes x", "a = 0..4; x = 2; for i in 0..3 { x += a; } return x;"),
    ("dice made in a loop", "s = 0; for i in 0..4 { d = 0..2; s += d; } return s;"),
    ("a walk", "a = 0..4; x = 0; for i in 0..6 { if i < 3 { x += a; } else { x -= 1; } } return x;"),
    ("nested loops with a condition",
     "a = 0..4; c = 0; for i in 0..3 { for j in 0..3 { if i < j { c += a; } } } return c;"),
    ("a condition on a distribution in a loop", "a = 0..4; s = 0; for i in 0..3 { if a < 2 { s += 1; } } return s;"),
    ("a loop variable compared with a distribution", "a = 0..4; s = 0; for i in 0..4 { if a < i { s += 1; } } return s;"),
    # functions
    ("a function copies an integer", "function f(p) { p += 3; return p; }\nx = 2; y = f(x); return x + y;"),
    ("a function shares a distribution", BUMP + "a = 0..4; b = bump(a); return a;"),
    ("a returned distribution is an alias", BUMP + "a = 0..4; b = bump(a); b += 10; return a;"),
    ("a computed return is independent", "function plus(p) { return p + 1; }\na = 0..4; c = plus(a); return c - a;"),
    ("a returned range is fresh", "function die() { return 0..4; }\nd = die(); return d;"),
    ("a function called in a loop",
     "function shift(p, k) { return p + k; }\na = 0..4; t = 0; for i in 0..4 { t += shift(a, i); } return t;"),
    ("a function with a condition on its argument",
     "function pick(p, n) { if n < 2 { p += 1; } return p; }\na = 0..4; r = pick(a, 1); q = pick(a, 5); return a;"),
    ("a call statement called twice", BUMP + "a = 0..4; bump(a); bump(a); return a;"),
    # a variable a region may superpose, then copied or aliased
    ("copy after a dead branch", "a = 0..4; x = 2; k = 9; for i in 0..2 { if k < 3 { x += a; } } y = x; y += 1; return x;"),
    ("alias after a live branch", "a = 0..4; x = 2; k = 1; for i in 0..2 { if k < 3 { x += a; } } y = x; y += 1; return x;"),
    ("update the original, read a dead copy",
     "a = 0..4; x = 2; k = 9; for i in 0..2 { if k < 3 { x += a; } } y = x; x += 1; return y;"),
    ("update the original, read a live alias",
     "a = 0..4; x = 2; k = 1; for i in 0..2 { if k < 3 { x += a; } } y = x; x += 1; return y;"),
    ("a chain of copies after a dead branch",
     "a = 0..4; x = 2; k = 9; for i in 0..2 { if k < 3 { x += a; } } y = x; z = y; z += 1; return x;"),
    ("a chain of aliases after a live branch",
     "a = 0..4; x = 2; k = 1; for i in 0..2 { if k < 3 { x += a; } } y = x; z = y; z += 1; return x;"),
    ("an assignment leaves the group",
     "a = 0..4; x = 2; k = 9; for i in 0..2 { if k < 3 { x += a; } } y = x; x = 7; y += 1; return x;"),
    ("a condition on a variable a dead branch could superpose",
     "a = 0..4; x = 2; y = 0; k = 9; for i in 0..3 { if k < 3 { x += a; } if x < 5 { y += 1; } } return y;"),
    ("a possibly superposed variable passed to a function",
     BUMP + "a = 0..4; x = 2; k = 9; for i in 0..2 { if k < 3 { x += a; } } y = bump(x); return x;"),
)


class Outcome(NamedTuple):
    status: str  # "ok", "WRONG", "refused" or "failed"
    detail: str = ""


def distance(counts: Dict[int, int], expected: Dict[int, float]) -> float:
    total = sum(counts.values())
    return 0.5 * sum(abs(counts.get(value, 0) / total - expected.get(value, 0)) for value in set(counts) | set(expected))


def check(source: str, expected: Dict[int, float], fold: bool) -> Outcome:
    try:
        text = translate(parse(source, "v2"), fold=fold)
    except TranslationError as error:
        return Outcome("refused", str(error))
    try:
        counts = simulate.run(text, shots=SHOTS)
    except Exception as error:  # the tool or the lowering stopped it
        message = (getattr(error, "stderr", "") or str(error)).strip().splitlines()
        return Outcome("refused", (message[0] if message else str(error))[:100])
    gap = distance(counts, expected)
    return Outcome("ok" if gap < TOLERANCE else "WRONG", f"distance {gap:.2f}, got {dict(sorted(counts.items()))}")


def random_program(rng: random.Random) -> str:
    """
    A small program aimed at the risky constructs: integers that a live or a dead branch (or a loop) may superpose, then
    copied, aliased, updated through either name and read. Some are not valid under the semantics (assigning over a
    distribution), and the reference rejects those, so they are skipped.
    """
    names = ["x", "y", "z"]

    def simple() -> str:
        v, w = rng.sample(names, 2)
        return rng.choice([f"{v} += a;", f"{v} += a;", f"{v} += {rng.randint(1, 3)};", f"{v} = {w};", f"{v} = {w};",
                           f"{v} = {rng.randint(1, 6)};", f"{v} += {w};"])

    def statement() -> str:
        body = " ".join(simple() for _ in range(rng.randint(1, 2)))
        kind = rng.choice(["plain", "plain", "live", "dead", "loop"])
        if kind == "live":
            return f"if kl < 3 {{ {body} }}"
        if kind == "dead":
            return f"if kd < 3 {{ {body} }}"
        return f"for i in 0..2 {{ {body} }}" if kind == "loop" else body

    lines = ["a = 0..4;", "x = 2;", "y = 3;", "z = 5;", "kl = 1;", "kd = 9;"]
    lines += [statement() for _ in range(rng.randint(3, 7))]
    lines.append(rng.choice(["return x;", "return y;", "return x - y;", "return x + z;", "return z - y;"]))
    return " ".join(lines)


def fuzz(count: int, seed: int) -> None:
    rng = random.Random(seed)
    tally: Dict[Tuple[str, str], int] = {}
    reasons: Dict[Tuple[str, str], int] = {}
    wrong: List[str] = []
    checked = 0
    while checked < count:
        source = random_program(rng)
        try:
            expected = law(parse(source, "v2"))
        except ReferenceError_:
            continue
        checked += 1
        for label, fold in (("A", False), ("B", True)):
            outcome = check(source, expected, fold)
            tally[(label, outcome.status)] = tally.get((label, outcome.status), 0) + 1
            if outcome.status == "refused":
                key = (label, outcome.detail[:90])
                reasons[key] = reasons.get(key, 0) + 1
            if outcome.status == "WRONG":
                wrong.append(f"{label}: {source}\n     {outcome.detail}  expected {expected}")
    print(f"{checked} random programs valid under the semantics (seed {seed})")
    for label in ("A", "B"):
        print(f"  {label}: " + ", ".join(f"{status} {tally.get((label, status), 0)}" for status in ("ok", "refused", "WRONG")))
    for (label, reason), times in sorted(reasons.items(), key=lambda item: -item[1])[:8]:
        print(f"    refused by {label} x{times}: {reason}")
    for text in wrong[:10]:
        print(text)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "fuzz":
        fuzz(int(sys.argv[2]) if len(sys.argv) > 2 else 200, int(sys.argv[3]) if len(sys.argv) > 3 else 1)
        return
    wrong: List[str] = []
    print(f"{'program':<56}{'reference':<22}{'A':<10}{'B':<10}")
    for name, source in PROGRAMS:
        try:
            expected = law(parse(source, "v2"))
        except ReferenceError_ as error:
            print(f"{name:<56}not modelled: {error}")
            continue
        shown = ", ".join(f"{value}:{chance:.2f}" for value, chance in sorted(expected.items()))
        a, b = check(source, expected, False), check(source, expected, True)
        print(f"{name:<56}{(shown if len(shown) < 21 else shown[:19] + '…'):<22}{a.status:<10}{b.status:<10}")
        for label, outcome in (("A", a), ("B", b)):
            if outcome.status == "WRONG":
                wrong.append(f"{name} ({label})")
                print(f"{'':<56}  {label}: {outcome.detail}  expected {shown}")
    print(f"\nwrong answers: {len(wrong)}" + ("" if not wrong else "  -> " + "; ".join(wrong)))


if __name__ == "__main__":
    main()

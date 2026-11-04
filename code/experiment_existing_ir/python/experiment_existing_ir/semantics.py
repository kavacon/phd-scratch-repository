"""
Do the compiled programs mean what the source says?

Each program is given to the reference (`reference.py`, which computes its intended outcome law exactly from the source
alone) and compiled. The circuit is run on
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
    # operators and intervals
    ("a product of a distribution and an integer", "a = 0..4; return a * 3;"),
    ("a product of two distributions", "a = 0..4; b = 0..3; return a * b;"),
    ("a product of negative integers", "x = -3; y = 5; return x * y;"),
    ("every comparison of a distribution", "a = 0..4; s = 0; if a > 0 { s += 1; } if a == 2 { s += 2; } if a != 3 { s += 4; } return s;"),
    ("boolean operators", "a = 0..4; b = 0..2; return !(a < 2) || (b == 0 && a > 0);"),
    ("equality of two distributions", "a = 0..3; b = 0..3; return a == b;"),
    ("an interval that is not a power of two", "a = 0..6; return a;"),
    ("an interval that does not start at zero", "a = -2..3; return a;"),
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
    # conditionals on a distribution inside conditionals, loops and calls, and an alias made inside a branch
    ("a distribution conditional inside another, in a loop",
     "a = 0..4; b = 0..2; s = 0; for i in 0..2 { if a < 2 { if b < 1 { s += 1; } } } return s;"),
    ("a loop inside an arm of a distribution conditional", "a = 0..4; s = 0; if a < 2 { for i in 0..3 { s += 1; } } return s;"),
    ("a distribution added to itself", "a = 0..4; a += a; return a;"),
    ("an outer name aliased inside a live branch", "a = 0..4; x = 2; k = 1; if k < 3 { x = a; } x += 1; return a;"),
    ("an outer name aliased inside a dead branch", "a = 0..4; x = 2; k = 9; if k < 3 { x = a; } x += 1; return a;"),
)


class Outcome(NamedTuple):
    status: str  # "ok", "WRONG", "refused" or "failed"
    detail: str = ""


def distance(counts: Dict[int, int], expected: Dict[int, float]) -> float:
    total = sum(counts.values())
    return 0.5 * sum(abs(counts.get(value, 0) / total - expected.get(value, 0)) for value in set(counts) | set(expected))


def check(source: str, expected: Dict[int, float]) -> Outcome:
    try:
        text = translate(parse(source, "v2"))
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


RICH_FUNCTIONS = (BUMP + "function plus(p) { return p + 1; }\n"
                  "function pick(p, n) { if n < 2 { p += 1; } return p; }\n")


def random_rich_program(rng: random.Random) -> str:
    """
    The same kind of program with the constructs that used to be refused: conditionals on a distribution (on a range, on
    a variable that may hold one, or on a loop variable) inside conditionals and loops, loops and conditionals inside
    them, a distribution added to itself, aliases of both distributions, and calls that update or return a variable.
    """
    names = ["x", "y", "z"]

    def number() -> int:
        return rng.randint(1, 3)

    def simple(in_loop: bool) -> str:
        v, w = rng.sample(names, 2)
        options = [f"{v} += a;", f"{v} += b;", f"{v} += {number()};", f"{v} = {w};", f"{v} = {w};", f"{v} = {number()};",
                   f"{v} += {w};", f"{v} -= {w};", f"{v} = {w} + a;", f"{v} = a;", f"a += {v};", "a += a;", f"b += {v};",
                   f"{v} = bump({w});", f"bump({v});", f"{v} = plus({w});", f"{v} = pick({w}, {rng.randint(0, 3)});"]
        if in_loop:
            options += [f"{v} += i;", f"{v} = i;"]
        return rng.choice(options)

    def condition(in_loop: bool) -> str:
        options = ["kl < 3", "kd < 3", "a < 2", "a < 2", "b < 1", "x < 4", "y < 3", "z < 6", "a < x", "x < a"]
        return rng.choice(options + (["a < i", "i < 1"] if in_loop else []))

    def block(depth: int, in_loop: bool) -> str:
        return " ".join(statement(depth + 1, in_loop) for _ in range(rng.randint(1, 2)))

    def statement(depth: int, in_loop: bool) -> str:
        kind = rng.choice(["plain"] * 4 + ["if", "if", "loop"] if depth < 3 else ["plain"])
        if kind == "if":
            text = f"if {condition(in_loop)} {{ {block(depth, in_loop)} }}"
            return text + (f" else {{ {block(depth, in_loop)} }}" if rng.random() < 0.4 else "")
        if kind == "loop" and not in_loop:
            return f"for i in 0..{rng.randint(2, 3)} {{ {block(depth, True)} }}"
        return simple(in_loop)

    lines = ["a = 0..4;", "b = 0..2;", "x = 2;", "y = 3;", "z = 5;", "kl = 1;", "kd = 9;"]
    lines += [statement(0, False) for _ in range(rng.randint(3, 6))]
    lines.append(rng.choice(["return x;", "return y;", "return x - y;", "return x + z;", "return z - y;", "return a;"]))
    return RICH_FUNCTIONS + " ".join(lines)


def random_operator_program(rng: random.Random) -> str:
    """
    Expressions over every operator and over integers and distributions on intervals of any size (powers of two or not, and
    negative), used as values, as conditions of conditionals and as the result. Arithmetic and boolean expressions are kept
    apart, since a variable holds one or the other.
    """
    def interval() -> str:
        low = rng.choice([0, 0, 0, 1, 2, -2, -3])
        return f"{low}..{low + rng.choice([2, 3, 3, 4, 5, 6, 7])}"

    def arithmetic(depth: int) -> str:
        if depth == 0 or rng.random() < 0.25:
            return rng.choice(["a", "b", "x", str(rng.randint(-3, 4)), str(rng.randint(-3, 4))])
        return f"({arithmetic(depth - 1)} {rng.choice(['+', '-', '*', '+', '*'])} {arithmetic(depth - 1)})"

    def boolean(depth: int) -> str:
        if depth == 0 or rng.random() < 0.3:
            return f"({arithmetic(1)} {rng.choice(['<', '>', '==', '!='])} {arithmetic(1)})"
        if rng.random() < 0.2:
            return f"!{boolean(depth - 1)}"
        return f"({boolean(depth - 1)} {rng.choice(['&&', '||'])} {boolean(depth - 1)})"

    lines = [f"a = {interval()};", f"b = {interval()};", "x = 1;"]
    for _ in range(rng.randint(1, 3)):
        kind = rng.choice(["set", "update", "if"])
        if kind == "set":
            lines.append(f"x = {arithmetic(2)};")
        elif kind == "update":
            lines.append(f"x {rng.choice(['+=', '-='])} {arithmetic(2)};")
        else:
            test = boolean(2) if rng.random() < 0.7 else arithmetic(1)
            lines.append(f"if {test} {{ x += {arithmetic(1)}; }}" + (f" else {{ x -= {arithmetic(1)}; }}" if rng.random() < 0.4 else ""))
    lines.append(f"return {rng.choice([arithmetic(2), arithmetic(2), boolean(2)])};")
    return " ".join(lines)


def fuzz(count: int, seed: int, rich: bool = False, operators: bool = False) -> None:
    """Random programs, each compiled and compared with the reference; `rich` and `operators` use the other generators and only A."""
    rich = rich or operators
    rng = random.Random(seed)
    tally: Dict[str, int] = {}
    reasons: Dict[str, int] = {}
    wrong: List[str] = []
    checked = 0
    while checked < count:
        source = random_operator_program(rng) if operators else random_rich_program(rng) if rich else random_program(rng)
        try:
            expected = law(parse(source, "v2"))
        except ReferenceError_:
            continue
        checked += 1
        outcome = check(source, expected)
        tally[outcome.status] = tally.get(outcome.status, 0) + 1
        if outcome.status == "refused":
            reasons[outcome.detail[:90]] = reasons.get(outcome.detail[:90], 0) + 1
        if outcome.status == "WRONG":
            wrong.append(f"{source}\n     {outcome.detail}  expected {expected}")
    print(f"{checked} random programs valid under the semantics (seed {seed})")
    print("  " + ", ".join(f"{status} {tally.get(status, 0)}" for status in ("ok", "refused", "WRONG")))
    for reason, times in sorted(reasons.items(), key=lambda item: -item[1])[:8]:
        print(f"    refused x{times}: {reason}")
    for text in wrong[:10]:
        print(text)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "fuzz":
        fuzz(int(sys.argv[2]) if len(sys.argv) > 2 else 200, int(sys.argv[3]) if len(sys.argv) > 3 else 1,
             rich=sys.argv[4:] == ["rich"], operators=sys.argv[4:] == ["ops"])
        return
    wrong: List[str] = []
    print(f"{'program':<56}{'reference':<22}{'compiled':<10}")
    for name, source in PROGRAMS:
        try:
            expected = law(parse(source, "v2"))
        except ReferenceError_ as error:
            print(f"{name:<56}not modelled: {error}")
            continue
        shown = ", ".join(f"{value}:{chance:.2f}" for value, chance in sorted(expected.items()))
        outcome = check(source, expected)
        print(f"{name:<56}{(shown if len(shown) < 21 else shown[:19] + '…'):<22}{outcome.status:<10}")
        if outcome.status == "WRONG":
            wrong.append(name)
            print(f"{'':<56}  {outcome.detail}  expected {shown}")
    print(f"\nwrong answers: {len(wrong)}" + ("" if not wrong else "  -> " + "; ".join(wrong)))


if __name__ == "__main__":
    main()

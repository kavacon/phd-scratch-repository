"""
The same programs through two ways of folding what is known at compile time, and what each gives.

A leaves it to MLIR: the translation keeps loops, conditionals and calls, a variable that may hold a distribution holds a
reference to an object, and `canonicalize`, `mem2reg` and `flatten-dist-if` resolve the rest.
B folds as it translates: known integers are constants, a conditional keeps only the arm taken, and loops and calls are
expanded in place, so nothing is left for the passes to decide.

For each program and each approach this reports whether it compiled, the outcome it gives (sampled on Aer, against the
chances worked out by hand), the registers and qubits, the operations before the lowering, and the circuit after it.

    . ./env.sh && .venv/bin/python -m experiment_existing_ir.comparison
"""
from typing import Dict, List, NamedTuple, Optional

from checking import qiskit_tools
from compiler.parser.parser import parse
from experiment_existing_ir import simulate
from experiment_existing_ir.translate import TranslationError, translate

SHOTS = 4000


class Program(NamedTuple):
    name: str
    source: str
    expected: Dict[int, float]  # the chance of each outcome, worked out by hand


def uniform(*values: int) -> Dict[int, float]:
    return {value: 1 / len(values) for value in values}


def convolve(laws: List[Dict[int, float]]) -> Dict[int, float]:
    total = {0: 1.0}
    for law in laws:
        combined: Dict[int, float] = {}
        for left, p in total.items():
            for right, q in law.items():
                combined[left + right] = combined.get(left + right, 0) + p * q
        total = combined
    return total


# a is 0..3, so 2 + a is 2..5; three tied copies of a make 3a
PROGRAMS = (
    Program("integers only: a loop doubling x eight times", "x = 1; for i in 0..8 { x += x; } return x;", {256: 1}),
    Program("a dead branch would turn x into a distribution",
            "a = 0..4; x = 2; k = 5; if k < 3 { x += a; } y = x + 1; return y;", {3: 1}),
    Program("a taken branch turns x into a distribution",
            "a = 0..4; x = 2; k = 1; if k < 3 { x += a; } return x;", uniform(2, 3, 4, 5)),
    Program("a walk: add a for three steps, then subtract one",
            "a = 0..4; x = 0; for i in 0..6 { if i < 3 { x += a; } else { x -= 1; } } return x;",
            uniform(-3, 0, 3, 6)),
    Program("dice made inside a loop", "s = 0; for i in 0..4 { d = 0..2; s += d; } return s;",
            convolve([uniform(0, 1)] * 4)),
    Program("pairs i < j in nested loops add a each",
            "a = 0..4; c = 0; for i in 0..3 { for j in 0..3 { if i < j { c += a; } } } return c;",
            uniform(0, 3, 6, 9)),
    Program("a function called in a loop",
            "function shift(p, k) { return p + k; }\na = 0..4; t = 0; for i in 0..4 { t += shift(a, i); } return t;",
            uniform(6, 10, 14, 18)),
    Program("a function with a conditional on its argument",
            "function pick(p, n) { if n < 2 { p += 1; } return p; }\na = 0..4; r = pick(a, 1); q = pick(a, 5); return a;",
            uniform(1, 2, 3, 4)),
    Program("a condition on a variable that a dead branch could superpose",
            "a = 0..4; x = 2; y = 0; k = 9; for i in 0..3 { if k < 3 { x += a; } if x < 5 { y += 1; } } return y;", {3: 1}),
    Program("copying a variable that a dead branch could superpose",
            "a = 0..4; x = 2; k = 9; for i in 0..2 { if k < 3 { x += a; } } y = x; y += 1; return x;", {2: 1}),
    Program("a condition on a distribution inside a loop",
            "a = 0..4; s = 0; for i in 0..3 { if a < 2 { s += 1; } } return s;", {0: 0.5, 3: 0.5}),
    Program("a loop variable compared with a distribution", "a = 0..4; s = 0; for i in 0..4 { if a < i { s += 1; } } return s;",
            uniform(0, 1, 2, 3)),
)


class Result(NamedTuple):
    outcome: str  # "ok", "off by <distance>", or why it did not compile
    qubits: Optional[int] = None
    registers: Optional[int] = None
    operations: Optional[int] = None  # operations of the program before the lowering
    gates: Optional[int] = None
    depth: Optional[int] = None


def distance(counts: Dict[int, int], expected: Dict[int, float]) -> float:
    """Total variation distance between the sampled outcomes and the chances worked out by hand."""
    total = sum(counts.values())
    return 0.5 * sum(abs(counts.get(value, 0) / total - expected.get(value, 0)) for value in set(counts) | set(expected))


def compile_with(program: Program, fold: bool) -> Result:
    try:
        text = translate(parse(program.source, "v2"), fold=fold)
        folded = simulate.optimise(text, *simulate.DEFAULT[:-1])
        qc = simulate.optimise(text, *simulate.DEFAULT)
    except TranslationError as error:
        return Result(f"refused: {error}")
    except Exception as error:  # the tool or the lowering failed
        return Result(f"failed: {(getattr(error, 'stderr', '') or str(error)).strip().splitlines()[0][:90]}")
    operations = sum(1 for line in folded.splitlines() if " = " in line or line.strip().startswith(("return", "scf.", "func.return")))
    counts = simulate.execute(qc, SHOTS)
    circuit = qiskit_tools.read(simulate.openqasm(qc))
    while subroutines(circuit):
        circuit = circuit.decompose(gates_to_decompose=subroutines(circuit))
    stats = qiskit_tools.measure(circuit)
    gap = distance(counts, program.expected)
    qubits = sum(int(line.split("memref<")[1].split("x")[0]) for line in qc.splitlines() if "memref.alloc" in line)
    return Result("ok" if gap < 0.05 else f"off by {gap:.2f}", qubits, qc.count("memref.alloc"), operations,
                  stats["gates"], stats["depth"])


def subroutines(circuit) -> List[str]:
    """Names of the adders the exported circuit calls, which count as the gates they are made of."""
    return [name for name in circuit.count_ops() if name.startswith("add")]


def main() -> None:
    columns = ("outcome", "qubits", "registers", "ir ops", "gates", "depth")
    print(f"{'program':<52}{'':<3}" + "".join(f"{c:>11}" for c in columns))
    for program in PROGRAMS:
        for label, fold in (("A", False), ("B", True)):
            result = compile_with(program, fold)
            cells = [result.outcome if len(result.outcome) < 10 else result.outcome[:9] + "…"] + [
                "-" if value is None else str(value) for value in result[1:]]
            print(f"{program.name if label == 'A' else '':<52}{label:<3}" + "".join(f"{c:>11}" for c in cells))
            if result.outcome not in ("ok",) and not result.outcome.startswith("off"):
                print(f"{'':<55}{result.outcome}")


if __name__ == "__main__":
    main()

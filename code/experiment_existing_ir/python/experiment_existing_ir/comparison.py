"""
Source-level rewrites against gate-level optimisation. Each program is compiled with the source rewrites and without them
(`pipeline.UNREWRITTEN`), each circuit is run through each optimiser (`optimisers`), and the result is measured in Clifford and T
(`metrics.of_clifford_t`). Every optimised circuit is sampled and must give the same outcomes as the program's unoptimised circuit.

    . ./env.sh && .venv/bin/python -m experiment_existing_ir.comparison [cache.json [program name filter]]

The programs are the measurement workloads (`workloads`): each holds a pattern the literature reports as costly or lossy.

Besides the two compilations above there is a ladder: straight lowering, then one source-level step added at a time (CSE, in-place
updates, the constant-step and zero rewrites, flattening, range folds, sharing between arms), to show which structure each step pays
off on. Each rung is also built with release and the qubit pool, to give the width the real pipeline would have.

Results are kept in the cache file as they are made, so a stopped run continues.
"""
import json
import os
import sys
import time
from typing import Dict, List, NamedTuple, Optional

from checking import qiskit_tools
from checking.circuits import to_signed
from compiler.parser.parser import parse
from experiment_existing_ir import metrics, optimisers, pipeline
from experiment_existing_ir.translate import translate
from experiment_existing_ir.workloads import WORKLOADS

SHOTS = 3000
TOLERANCE = 0.07  # total variation distance allowed between two samplings of the same outcomes
REWRITES = {"rewrites": optimisers.unpooled(pipeline.DEFAULT), "no rewrites": optimisers.unpooled(pipeline.UNREWRITTEN)}
TAIL = ("--dist-status", "--plan-registers", "--schedule-release{strategy=lazy}", "--lower-to-qc", "--release-registers", *pipeline.UNROLL_MODIFIERS)


def rung(cse=False, in_place=False, steps=False, flatten=False, ranges=False, share=False):
    """The stages of one rung of the ladder (before the pool)."""
    stages = [*pipeline.PREPARE, "--resolve-dist", "--sccp", "--canonicalize"]
    if cse:
        stages.append("--cse")
    stages.append("--resolve-instances" if share else "--resolve-instances{share-arms=false}")
    if steps or flatten:
        stages.append("--dist-rewrites{combine-steps=%s flatten=%s}" % (str(steps).lower(), str(flatten).lower()))
    stages.append("--canonicalize")
    if ranges:
        stages += ["--fold-by-range", "--canonicalize"]
    if cse:
        stages.append("--cse")
    if in_place:
        stages.append("--in-place-updates")
    return (*stages, *TAIL)


LADDER = {"straight lowering": rung(), "+ CSE": rung(True), "+ in-place updates": rung(True, True),
          "+ constant steps & zero": rung(True, True, True), "+ flatten nested ifs": rung(True, True, True, True),
          "+ range folds": rung(True, True, True, True, True), "+ arm sharing": rung(True, True, True, True, True, True)}
PIPELINES = {**REWRITES, **LADDER}


class Cell(NamedTuple):
    program: str
    label: str  # which compilation: see PIPELINES
    optimiser: str
    cost: Optional[list]  # metrics.TCost, or None when the optimiser did not finish
    seconds: float
    error: float  # distance from the unoptimised circuit's outcomes
    leaked: int  # qubits left set that were not left set without the optimiser
    pooled: int  # qubits of the circuit with release and the qubit pool, before any optimiser


def programs() -> Dict[str, str]:
    return {w[1]: w[2] for w in WORKLOADS}


def distance(counts: Dict[int, int], expected: Dict[int, float]) -> float:
    total = sum(counts.values())
    return 0.5 * sum(abs(counts.get(value, 0) / total - expected.get(value, 0)) for value in set(counts) | set(expected))


def outcomes(circuit):
    width = circuit.num_clbits
    result = qiskit_tools.execute(circuit, width, shots=SHOTS)
    return {to_signed(value, width): n for value, n in result["counts"].items()}, len(result["leaked"])


MEMO: Dict[tuple, tuple] = {}  # an optimiser's result on a circuit it has already seen (steps that leave a program as it was)


def measure(name: str, source: str, label: str) -> List[Cell]:
    """Every optimiser on the program compiled one way."""
    program = translate(parse(source, "v2"))
    qc = pipeline.optimise(program, *PIPELINES[label])
    pooled = metrics.of_qc(pipeline.optimise(program, *PIPELINES[label], "--assign-qubits")).qubits
    reference, held = outcomes(optimisers.none(qc))
    cells = []
    for optimiser, run in optimisers.OPTIMISERS.items():
        start = time.time()
        if (optimiser, qc) not in MEMO:
            try:
                circuit = run(qc)
            except optimisers.TimedOut:
                MEMO[(optimiser, qc)] = (None, 0.0, 0)
            else:
                counts, leaked = outcomes(circuit)
                MEMO[(optimiser, qc)] = (list(metrics.of_clifford_t(circuit)), distance(counts, {k: v / SHOTS for k, v in reference.items()}), leaked - held)
        cost, error, leaked = MEMO[(optimiser, qc)]
        cells.append(Cell(name, label, optimiser, cost, time.time() - start, error, leaked, pooled))
    return cells


def run(cache: Optional[str] = None, only: str = "") -> List[Cell]:
    done = [Cell(*row) for row in json.load(open(cache))] if cache and os.path.exists(cache) else []
    seen = {(c.program, c.label) for c in done}
    for name, source in programs().items():
        if only not in name:
            continue
        for label in PIPELINES:
            if (name, label) in seen:
                continue
            done.extend(measure(name, source, label))
            if cache:
                json.dump(done, open(cache, "w"))
            print(name, label, "done", flush=True)
    return done


def report(cells: List[Cell]) -> None:
    """Totals over the programs every optimiser finished, so that the columns are over the same programs."""
    by = {(c.program, c.label, c.optimiser): c for c in cells}
    names = sorted({c.program for c in cells})
    names = [n for n in names if all(by.get((n, r, o)) and by[(n, r, o)].cost for r in REWRITES for o in optimisers.OPTIMISERS)]
    skipped = sorted({c.optimiser for c in cells if c.cost is None})
    print(f"{len(names)} programs finished by every optimiser; timeouts: {skipped or 'none'}; "
          f"outcomes differing from the unoptimised circuit by more than {TOLERANCE}: "
          f"{sum(1 for c in cells if c.error > TOLERANCE)}; qubits left set that were not before: {sum(1 for c in cells if c.leaked > 0)}\n")
    for index, label in ((4, "T count"), (5, "T depth"), (2, "two-qubit gates"), (3, "depth"), (1, "gates")):
        print(f"{label}:")
        print(f"  {'optimiser':<10}{'no rewrites':>13}{'rewrites':>10}{'rewrites/no':>13}")
        base = sum(by[(n, 'no rewrites', 'none')].cost[index] for n in names)
        for optimiser in optimisers.OPTIMISERS:
            off = sum(by[(n, 'no rewrites', optimiser)].cost[index] for n in names)
            on = sum(by[(n, 'rewrites', optimiser)].cost[index] for n in names)
            print(f"  {optimiser:<10}{off:>13}{on:>10}{on / off:>13.2f}   (optimiser alone: {off / base:.2f} of the unoptimised, unrewritten)")
        print()


FAMILY = {w[1]: w[0] for w in WORKLOADS}
METRICS = ((4, "T count"), (5, "T depth"), (2, "two-qubit gates"), (1, "gates"), (3, "depth"), (0, "qubits (a wire for every register)"))


def report_ladder(cells: List[Cell]) -> None:
    """Each rung of the ladder under each optimiser, totalled over the programs every optimiser finished at every rung."""
    by = {(c.program, c.label, c.optimiser): c for c in cells}
    names = [n for n in sorted({c.program for c in cells})
             if all(by.get((n, r, o)) and by[(n, r, o)].cost for r in LADDER for o in optimisers.OPTIMISERS)]
    print(f"ladder, over {len(names)} programs finished by every optimiser at every rung\n")
    for index, label in METRICS:
        print(f"{label}:")
        print(f"  {'':<26}" + "".join(f"{o:>9}" for o in optimisers.OPTIMISERS))
        for rung_label in LADDER:
            print(f"  {rung_label:<26}" + "".join(f"{sum(by[(n, rung_label, o)].cost[index] for n in names):>9}" for o in optimisers.OPTIMISERS))
        print()
    print("qubits with release and the qubit pool (no optimiser):")
    for rung_label in LADDER:
        print(f"  {rung_label:<26}{sum(by[(n, rung_label, 'none')].pooled for n in names):>9}")
    for optimiser in ("none", "mqt", "pyzx"):
        print(f"\nT count by family, {optimiser}:")
        print(f"  {'':<26}" + "".join(f"{f:>8}" for f in ("nest", "late", "arms", "chain", "loss", "calls")))
        for rung_label in LADDER:
            row = [sum(by[(n, rung_label, optimiser)].cost[4] for n in names if FAMILY[n] == f) for f in ("nest", "late", "arms", "chain", "loss", "calls")]
            print(f"  {rung_label:<26}" + "".join(f"{x:>8}" for x in row))


def main() -> None:
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    cells = run(cache, sys.argv[2] if len(sys.argv) > 2 else "")
    report(cells)
    report_ladder(cells)


if __name__ == "__main__":
    main()

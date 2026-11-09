"""
Source-level rewrites against gate-level optimisation. Each program is compiled with the source rewrites and without them
(`pipeline.UNREWRITTEN`), each circuit is run through each optimiser (`optimisers`), and the result is measured in Clifford and T
(`metrics.of_clifford_t`). Every optimised circuit is sampled and must give the same outcomes as the program's unoptimised circuit.

    . ./env.sh && .venv/bin/python -m experiment_existing_ir.comparison [cache.json [program name filter]]

The programs are the measurement workloads (`workloads`): each holds a pattern the literature reports as costly or lossy.

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



class Cell(NamedTuple):
    program: str
    rewrites: str
    optimiser: str
    cost: Optional[list]  # metrics.TCost, or None when the optimiser did not finish
    seconds: float
    error: float  # distance from the unoptimised circuit's outcomes
    leaked: int  # qubits left set that were not left set without the optimiser


def programs() -> Dict[str, str]:
    return {w[1]: w[2] for w in WORKLOADS}


def distance(counts: Dict[int, int], expected: Dict[int, float]) -> float:
    total = sum(counts.values())
    return 0.5 * sum(abs(counts.get(value, 0) / total - expected.get(value, 0)) for value in set(counts) | set(expected))


def outcomes(circuit):
    width = circuit.num_clbits
    result = qiskit_tools.execute(circuit, width, shots=SHOTS)
    return {to_signed(value, width): n for value, n in result["counts"].items()}, len(result["leaked"])


def measure(name: str, source: str, rewrites: str) -> List[Cell]:
    """Every optimiser on the program compiled one way."""
    qc = pipeline.optimise(translate(parse(source, "v2")), *REWRITES[rewrites])
    reference, held = outcomes(optimisers.none(qc))
    cells = []
    for optimiser, run in optimisers.OPTIMISERS.items():
        start = time.time()
        try:
            circuit = run(qc)
        except optimisers.TimedOut:
            cells.append(Cell(name, rewrites, optimiser, None, time.time() - start, 0.0, 0))
            continue
        seconds = time.time() - start
        counts, leaked = outcomes(circuit)
        cells.append(Cell(name, rewrites, optimiser, list(metrics.of_clifford_t(circuit)), seconds, distance(counts, {k: v / SHOTS for k, v in reference.items()}), leaked - held))
    return cells


def run(cache: Optional[str] = None, only: str = "") -> List[Cell]:
    done = [Cell(*row) for row in json.load(open(cache))] if cache and os.path.exists(cache) else []
    seen = {(c.program, c.rewrites) for c in done}
    for name, source in programs().items():
        if only not in name:
            continue
        for rewrites in REWRITES:
            if (name, rewrites) in seen:
                continue
            done.extend(measure(name, source, rewrites))
            if cache:
                json.dump(done, open(cache, "w"))
            print(name, rewrites, "done", flush=True)
    return done


def report(cells: List[Cell]) -> None:
    """Totals over the programs every optimiser finished, so that the columns are over the same programs."""
    by = {(c.program, c.rewrites, c.optimiser): c for c in cells}
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


def main() -> None:
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    report(run(cache, sys.argv[2] if len(sys.argv) > 2 else ""))


if __name__ == "__main__":
    main()

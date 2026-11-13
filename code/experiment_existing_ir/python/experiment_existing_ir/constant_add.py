"""
Adding a constant in place: the adder or the increment network, and what decides between them.

The adder (Cuccaro) needs a copy of the constant and a carry, so 2w + 1 qubits for a w-bit value, and its cost depends on the width only.
The increment network (`plan-registers{increment=true}`) needs no other qubit, and its cost depends on the width and on the bits of the
constant, which it writes as set bits or as signed digits (7 is 8 - 1), whichever is cheaper. Both are known before any gate is emitted,
so the choice is made from the source: `pipeline.with_auto_constant_add` (the `constant-add` pass) uses the network where the adder does not
fit and otherwise where its predicted cost, from a table of what an m-controlled X costs once decomposed (cpp/ConstantAdd.h, which a test
compares with `table`), is no worse than the adder's on two-qubit gates and on depth. This module measures the adder, the network and the
automatic choice for every constant, and how well the prediction does.

    . ./env.sh && .venv/bin/python -m experiment_existing_ir.constant_add [largest constant]

A multi-controlled X is decomposed by Qiskit (no ancilla given, but it uses idle qubits when the circuit has some), so the table, and
so the prediction, depends on that decomposition: it is a named assumption, not a property of the network.
"""
import sys
from typing import Dict, List, NamedTuple, Tuple

from qiskit import QuantumCircuit, transpile

from compiler.parser.parser import parse
from experiment_existing_ir import metrics, optimisers, pipeline
from experiment_existing_ir.translate import translate


class Cost(NamedTuple):
    two_qubit: int
    depth: int
    gates: int
    rotations: int


def mcx_cost(controls: int) -> Cost:
    """What an X with this many controls costs after decomposition, on a circuit with no other qubit."""
    circuit = QuantumCircuit(controls + 1)
    circuit.mcx(list(range(controls)), controls) if controls else circuit.x(0)
    cost = metrics.of_clifford_t(transpile(circuit, basis_gates=optimisers.BASIS, optimization_level=0))
    return Cost(cost.two_qubit, cost.depth, cost.gates, cost.rotations)


TABLE: Dict[int, Cost] = {}


def table(controls: int) -> Cost:
    if controls not in TABLE:
        TABLE[controls] = mcx_cost(controls)
    return TABLE[controls]


def network(width: int, constant: int, start: int = 0) -> List[int]:
    """The control counts of the gates that add `constant` to a register of `width` bits by increments from bit `start`."""
    controls = []
    for k in range(width):
        if (constant >> k) & 1:
            controls += [j - k for j in range(width - 1, k, -1)] + [0]
    return controls


def _sum(costs: List[Cost]) -> Cost:
    return Cost(sum(c.two_qubit for c in costs), sum(c.depth for c in costs), sum(c.gates for c in costs), sum(c.rotations for c in costs))


def step_costs(width: int, bit: int) -> List[Cost]:
    """The gates of the increment from `bit`: the bits above it from the top down, then the bit itself (a decrement is the same gates reversed)."""
    return [table(j - bit) for j in range(width - 1, bit, -1)] + [table(0)]


def predicted_plain(width: int, constant: int) -> Cost:
    """The network written as the constant's set bits: the gates run one after another (they share qubits), so depth adds."""
    return _sum([c for k in range(width) if (constant >> k) & 1 for c in step_costs(width, k)])


def predicted_signed(width: int, constant: int) -> Cost:
    return _sum([c for k, _ in signed_digits(constant, width) if k < width for c in step_costs(width, k)])


def predicted(width: int, constant: int) -> Cost:
    """The network's cost as the compiler chooses to write it: the cheaper of the two, fewest two-qubit gates, then least depth."""
    plain, signed = predicted_plain(width, constant), predicted_signed(width, constant)
    return signed if (signed.two_qubit, signed.depth) < (plain.two_qubit, plain.depth) else plain


def signed_digits(constant: int, width: int) -> List[Tuple[int, int]]:
    """The constant (mod 2^width) as the fewest terms +-2^k: 7 is 8 - 1. A decrement is an increment run backwards, so it costs the same."""
    constant %= 2 ** width
    digits, k = [], 0
    while constant:
        if constant & 1:
            digit = 2 - (constant & 3)
            digits.append((k, digit))
            constant -= digit
        constant >>= 1
        k += 1
    return digits


def measure(constant: int) -> Tuple[int, Cost, Cost, Cost, int, int, int]:
    """(width, adder, network, automatic choice, and the qubits of each) for `a += constant` on a value of two numbers, from the compiler."""
    program = translate(parse(f"a = 0..2; a += {constant}; return a;", "v2"))
    out = []
    for stages in (pipeline.DEFAULT, pipeline.with_increment(pipeline.DEFAULT), pipeline.with_auto_constant_add(pipeline.DEFAULT)):
        cost = metrics.of_clifford_t(optimisers.none(pipeline.optimise(program, *optimisers.unpooled(stages))))
        out.append((Cost(cost.two_qubit, cost.depth, cost.gates, cost.rotations), cost.qubits, metrics.of_qc(pipeline.optimise(program, *stages)).qubits))
    (adder, _, adder_qubits), (network, width, network_qubits), (auto, _, auto_qubits) = out
    return width, adder, network, auto, adder_qubits, network_qubits, auto_qubits


def main() -> None:
    largest = int(sys.argv[1]) if len(sys.argv) > 1 else 63
    rows = [(c, *measure(c)) for c in range(1, largest + 1)]  # constant, width, adder, network, automatic, and the qubits of each
    ADDER, NETWORK, AUTO = 2, 3, 4
    print(f"{len(rows)} constants, 1 to {largest}; widths {sorted({r[1] for r in rows})}")
    print("\nthe adder's cost depends on the width only (two-qubit gates, depth):",
          {w: sorted({(r[ADDER].two_qubit, r[ADDER].depth) for r in rows if r[1] == w}) for w in sorted({r[1] for r in rows})})
    print("\nthe network against the adder, by width (constants where the network is no worse on two-qubit gates and on depth, of all):")
    for w in sorted({r[1] for r in rows}):
        of = [r for r in rows if r[1] == w]
        both = [r for r in of if r[NETWORK].two_qubit <= r[ADDER].two_qubit and r[NETWORK].depth <= r[ADDER].depth]
        gates = [r for r in of if r[NETWORK].two_qubit <= r[ADDER].two_qubit]
        deep = [r for r in of if r[NETWORK].depth <= r[ADDER].depth]
        print(f"  width {w}: both {len(both)}/{len(of)}   two-qubit {len(gates)}/{len(of)}   depth {len(deep)}/{len(of)}")
    d2 = [r[NETWORK].two_qubit - r[ADDER].two_qubit for r in rows]
    dd = [r[NETWORK].depth - r[ADDER].depth for r in rows]
    mean2, meand = sum(d2) / len(d2), sum(dd) / len(dd)
    corr = sum((a - mean2) * (b - meand) for a, b in zip(d2, dd)) / (sum((a - mean2) ** 2 for a in d2) ** 0.5 * sum((b - meand) ** 2 for b in dd) ** 0.5)
    disagree = sum(1 for a, b in zip(d2, dd) if (a < 0) != (b < 0))
    print(f"\nnetwork minus adder: two-qubit gates against depth, correlation {corr:.2f}; they disagree on the sign for {disagree} of {len(rows)} constants")
    exact = sum(1 for r in rows if predicted(r[1], r[0]).two_qubit == r[NETWORK].two_qubit)
    within = sum(1 for r in rows if abs(predicted(r[1], r[0]).two_qubit - r[NETWORK].two_qubit) <= 0.1 * r[NETWORK].two_qubit)
    print(f"\nnetwork two-qubit gates predicted from (width, constant): exact for {exact}, within 10 percent for {within}, of {len(rows)}")
    recoded = [x for x in ((r[0], predicted_plain(r[1], r[0]), predicted(r[1], r[0])) for r in rows) if x[2].two_qubit < x[1].two_qubit]
    print(f"signed digits are used for {len(recoded)} of {len(rows)} constants; for those the predicted network goes from "
          f"{sum(x[1].two_qubit for x in recoded)} to {sum(x[2].two_qubit for x in recoded)} two-qubit gates, for example "
          f"{[(c, p.two_qubit, s.two_qubit) for c, p, s in recoded[:4]]}")
    print("\npolicies over all the constants (total two-qubit gates, total depth, total pooled qubits), all measured on the compiler:")
    for name, cost, qubits in (("always the adder", ADDER, 5), ("always the network (recoded)", NETWORK, 6), ("automatic (the constant-add pass)", AUTO, 7)):
        print(f"  {name:<40} {sum(r[cost].two_qubit for r in rows):>7} {sum(r[cost].depth for r in rows):>7} {sum(r[qubits] for r in rows):>5}")
    best = (sum(min(r[ADDER].two_qubit, r[NETWORK].two_qubit) for r in rows), sum(min(r[ADDER].depth, r[NETWORK].depth) for r in rows))
    print(f"  {'best of the two on each metric alone (a bound)':<40} {best[0]:>7} {best[1]:>7}")
    wrong = [(r[0], r[1]) for r in rows if (r[AUTO].two_qubit, r[AUTO].depth) != (r[ADDER].two_qubit, r[ADDER].depth) and
             (r[AUTO].two_qubit, r[AUTO].depth) != (r[NETWORK].two_qubit, r[NETWORK].depth)]
    worse = [r[0] for r in rows if r[AUTO].two_qubit > min(r[ADDER].two_qubit, r[NETWORK].two_qubit) * 1.1]
    print(f"\nthe automatic choice is more than 10 percent above the better of the two for {len(worse)} constants: {worse[:12]}")


if __name__ == "__main__":
    main()

"""
Adding a constant in place: the adder or the increment network, and what decides between them.

The adder (Cuccaro) needs a copy of the constant and a carry, so 2w + 1 qubits for a w-bit value, and its cost depends on the width only.
The increment network (`plan-registers{increment=true}`) needs no other qubit, and its cost depends on the width and on the bits of the
constant. Both are known before any gate is emitted, so the choice can be made from the source. This module measures both lowerings for
every constant, predicts the network's cost from (width, constant) with a table of what an m-controlled X costs once it is decomposed,
and tests the rule "use the network when it is no worse on two-qubit gates and on depth" against fixed choices.

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


def predicted(width: int, constant: int) -> Cost:
    """The network's cost from (width, constant) alone: the gates run one after another (they share qubits), so depth adds."""
    costs = [table(m) for m in network(width, constant)]
    return Cost(sum(c.two_qubit for c in costs), sum(c.depth for c in costs), sum(c.gates for c in costs), sum(c.rotations for c in costs))


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


def predicted_signed(width: int, constant: int) -> Cost:
    costs = [table(m) for k, _ in signed_digits(constant, width) for m in [j - k for j in range(width - 1, k, -1)] + [0]]
    return Cost(sum(c.two_qubit for c in costs), sum(c.depth for c in costs), sum(c.gates for c in costs), sum(c.rotations for c in costs))


def measure(constant: int) -> Tuple[int, Cost, Cost, int, int]:
    """(width, adder, network, adder qubits, network qubits) for `a += constant` on a value of two numbers, from the compiler."""
    program = translate(parse(f"a = 0..2; a += {constant}; return a;", "v2"))
    out = []
    for stages in (pipeline.DEFAULT, pipeline.with_increment(pipeline.DEFAULT)):
        cost = metrics.of_clifford_t(optimisers.none(pipeline.optimise(program, *optimisers.unpooled(stages))))
        out.append((cost, metrics.of_qc(pipeline.optimise(program, *stages)).qubits))
    (adder, adder_qubits), (increment, increment_qubits) = out
    return increment.qubits, Cost(adder.two_qubit, adder.depth, adder.gates, adder.rotations), \
        Cost(increment.two_qubit, increment.depth, increment.gates, increment.rotations), adder_qubits, increment_qubits


def main() -> None:
    largest = int(sys.argv[1]) if len(sys.argv) > 1 else 63
    rows = [(c, *measure(c)) for c in range(1, largest + 1)]
    print(f"{len(rows)} constants, 1 to {largest}; widths {sorted({r[1] for r in rows})}")
    print("\nthe adder's cost depends on the width only (two-qubit gates, depth):",
          {w: sorted({(r[2].two_qubit, r[2].depth) for r in rows if r[1] == w}) for w in sorted({r[1] for r in rows})})
    print("\nthe network against the adder, by width (constants where the network is no worse on two-qubit gates and on depth, of all):")
    for w in sorted({r[1] for r in rows}):
        of = [r for r in rows if r[1] == w]
        both = [r for r in of if r[3].two_qubit <= r[2].two_qubit and r[3].depth <= r[2].depth]
        gates = [r for r in of if r[3].two_qubit <= r[2].two_qubit]
        deep = [r for r in of if r[3].depth <= r[2].depth]
        print(f"  width {w}: both {len(both)}/{len(of)}   two-qubit {len(gates)}/{len(of)}   depth {len(deep)}/{len(of)}")
    # how closely do the two metrics move together?
    d2 = [r[3].two_qubit - r[2].two_qubit for r in rows]
    dd = [r[3].depth - r[2].depth for r in rows]
    mean2, meand = sum(d2) / len(d2), sum(dd) / len(dd)
    corr = sum((a - mean2) * (b - meand) for a, b in zip(d2, dd)) / (sum((a - mean2) ** 2 for a in d2) ** 0.5 * sum((b - meand) ** 2 for b in dd) ** 0.5)
    disagree = sum(1 for a, b in zip(d2, dd) if (a < 0) != (b < 0))
    print(f"\nnetwork minus adder: two-qubit gates against depth, correlation {corr:.2f}; they disagree on the sign for {disagree} of {len(rows)} constants")
    # the cost model against the compiler
    exact = sum(1 for r in rows if predicted(r[1], r[0]).two_qubit == r[3].two_qubit)
    within = sum(1 for r in rows if abs(predicted(r[1], r[0]).two_qubit - r[3].two_qubit) <= 0.1 * r[3].two_qubit)
    print(f"\nnetwork two-qubit gates predicted from (width, constant): exact for {exact}, within 10 percent for {within}, of {len(rows)}")

    def rule(width, constant, adder, model):
        net = model(width, constant)
        return net.two_qubit <= adder.two_qubit and net.depth <= adder.depth      # True: use the network
    print("\npolicies over all the constants (total two-qubit gates, total depth, total pooled qubits):")
    policies = {
        "always the adder": lambda r: False,
        "always the network": lambda r: True,
        "auto: network when predicted no worse on both": lambda r: rule(r[1], r[0], r[2], predicted),
        "auto, constant recoded in signed digits (prediction only)": None,
    }
    for name, choose in policies.items():
        if choose is None:
            continue
        pick = [(r[3] if choose(r) else r[2]) for r in rows]
        qubits = sum((r[5] if choose(r) else r[4]) for r in rows)
        print(f"  {name:<58} {sum(p.two_qubit for p in pick):>7} {sum(p.depth for p in pick):>7} {qubits:>5}")
    best = (sum(min(r[2].two_qubit, r[3].two_qubit) for r in rows), sum(min(r[2].depth, r[3].depth) for r in rows))
    print(f"  {'best of the two on each metric alone (a bound)':<58} {best[0]:>7} {best[1]:>7}")
    recoded = [(r[0], predicted(r[1], r[0]), predicted_signed(r[1], r[0])) for r in rows]
    cut = [x for x in recoded if x[2].two_qubit < x[1].two_qubit]
    print(f"\nsigned-digit recoding would cut the network's two-qubit gates for {len(cut)} of {len(rows)} constants; its total goes "
          f"{sum(x[1].two_qubit for x in recoded)} to {sum(x[2].two_qubit for x in recoded)} (predicted), depth "
          f"{sum(x[1].depth for x in recoded)} to {sum(x[2].depth for x in recoded)}; for example {[(c, p.two_qubit, s.two_qubit) for c, p, s in cut[:4]]}")


if __name__ == "__main__":
    main()

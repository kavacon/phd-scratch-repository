import itertools
import re
import unittest
from typing import Dict, FrozenSet, List, Optional, Tuple

from checking import qiskit_tools
from compiler.parser.parser import parse
from experiment_existing_ir import pipeline
from experiment_existing_ir.translate import translate
from tests.golden import GOLDEN

TOLERANCE = 1e-9
MOST_VALUE_REGISTERS = 14  # the search for the finest split tries every split of the registers whose value varies


def reported_groups(program: str) -> Dict[int, Optional[int]]:
    """
    What `dist-status` reports: for each register of the plan, the group of the last value held in it (None if definite), for the
    registers up to the measurement. A read gives a definite integer, and the measurement is the program's return, so what comes
    after it (widening the result for return) is output, which the circuit still holds as a correlated register.
    """
    text = pipeline.optimise(program, *pipeline.PLANNED)
    groups: Dict[int, Optional[int]] = {}
    for line in text.splitlines():
        register = re.search(r"existing_ir\.register = array<i64: (\d+)>", line)
        if register:
            group = re.search(r"dist\.group = (\d+)", line)
            groups[int(register.group(1))] = int(group.group(1)) if group else groups.get(int(register.group(1)))
        if "dist.measure" in line:
            break
    return groups


def distribution(circuit, interest: List[int]) -> Dict[int, float]:
    """
    The probability of each state of the given wires at the end of the circuit (bit k of the state is the k-th wire), without its
    measurement, by local simulation. Only the wires that vary are listed, so the table stays small however many qubits the
    circuit holds; a wire that never varies is set in every state or none.
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    run = circuit.remove_final_measurements(inplace=False)
    used = sorted({run.find_bit(q).index for i in run.data for q in i.qubits} | set(interest))
    position = {q: k for k, q in enumerate(used)}
    reduced = QuantumCircuit(len(used))
    for instruction in run.data:
        if instruction.operation.name != "barrier":
            reduced.append(instruction.operation, [position[run.find_bit(q).index] for q in instruction.qubits])
    runnable = transpile(reduced, basis_gates=qiskit_tools.BASIS, optimization_level=0)
    simulator = AerSimulator(method="matrix_product_state")
    probe = runnable.copy()
    for wire in interest:
        probe.save_probabilities(qubits=[position[wire]], label=f"q{wire}")
    saved = simulator.run(probe).result().data(0)
    chance = {wire: saved[f"q{wire}"][1] for wire in interest}
    varying = [wire for wire in interest if 1e-9 < chance[wire] < 1 - 1e-9]
    fixed = sum(1 << k for k, wire in enumerate(interest) if chance[wire] >= 1 - 1e-9)
    if not varying:
        return {fixed: 1.0}
    runnable.save_probabilities_dict(qubits=[position[wire] for wire in varying])
    states = simulator.run(runnable).result().data(0)["probabilities"]
    table = {}
    for state, p in states.items():
        if p > 1e-9:
            full = fixed | sum(((state >> j) & 1) << interest.index(wire) for j, wire in enumerate(varying))
            table[full] = p
    return table


def final_state(program: str) -> Tuple[Dict[int, List[int]], Dict[int, float], List[int]]:
    """
    The wires of each value register and the final distribution over all of them. The pool pass records on `main` which pool qubits hold
    each value register of the plan, and the exported circuit is one register of the pool's qubits, so a pool qubit is a wire.
    """
    qc = pipeline.build(program, *pipeline.DEFAULT)
    circuit = qiskit_tools.read(pipeline.export(qc, "openqasm3"))
    assert len(circuit.qregs) == 1, "the exported circuit is the pool"
    qubits = {int(number): [int(q) for q in wires.split(",")]
              for number, wires in re.findall(r"existing_ir\.qubits_of_(\d+) = array<i64: ([\d, ]+)>", qc)}
    wires = sorted({wire for register in qubits.values() for wire in register})
    return qubits, distribution(circuit, wires), wires


def marginal(states: Dict[int, float], wires: List[int], qubits: Dict[int, List[int]], registers: FrozenSet[int]):
    """The probability of each combination of values of the registers (a register's value is its qubits as an integer)."""
    position = {wire: k for k, wire in enumerate(wires)}
    result: Dict[Tuple[int, ...], float] = {}
    for state, p in states.items():
        key = tuple(sum(((state >> position[q]) & 1) << j for j, q in enumerate(qubits[r]) if q in position) for r in sorted(registers))
        result[key] = result.get(key, 0.0) + p
    return result


def true_groups(states, wires, qubits) -> List[FrozenSet[int]]:
    """The finest split of the registers into parts whose values are independent: the registers that depend on each other stay together."""
    def varies(register: int) -> bool:
        return len(marginal(states, wires, qubits, frozenset([register]))) > 1

    registers = [r for r in sorted(qubits) if varies(r)]
    assert len(registers) <= MOST_VALUE_REGISTERS, f"{len(registers)} registers vary"

    def split(group: List[int]) -> List[FrozenSet[int]]:
        if len(group) < 2:
            return [frozenset(group)]
        whole = marginal(states, wires, qubits, frozenset(group))
        order = {r: k for k, r in enumerate(sorted(group))}
        first, rest = group[0], group[1:]
        for size in range(len(rest)):
            for others in itertools.combinations(rest, size):
                part = frozenset((first, *others))
                other = frozenset(group) - part
                a, b = marginal(states, wires, qubits, part), marginal(states, wires, qubits, other)

                def project(key, registers):
                    return tuple(key[order[r]] for r in sorted(registers))

                # the joint is the product of its parts when every combination of their values has exactly the product's probability
                if len(whole) == len(a) * len(b) and all(
                        abs(p - a[project(key, part)] * b[project(key, other)]) < TOLERANCE for key, p in whole.items()):
                    return split(sorted(part)) + split(sorted(other))
        return [frozenset(group)]

    return split(registers)


class TrueGroups(unittest.TestCase):
    def test_registers_are_one_group_exactly_when_their_joint_distribution_is_not_a_product(self):
        qubits = {0: [0], 1: [1], 2: [2]}
        wires = [0, 1, 2]
        # register 0 and 1 are each a fair coin and equal to each other; register 2 is a fair coin on its own
        equal_pair = {0b000: 0.25, 0b011: 0.25, 0b100: 0.25, 0b111: 0.25}
        self.assertEqual(sorted(map(sorted, true_groups(equal_pair, wires, qubits))), [[0, 1], [2]])
        # three fair coins that are independent
        independent = {state: 1 / 8 for state in range(8)}
        self.assertEqual(sorted(map(sorted, true_groups(independent, wires, qubits))), [[0], [1], [2]])


@unittest.skipUnless(pipeline.built(), "existing-ir-opt is not built: run helper/cpp/setup.sh and cpp/build.sh")
class Groups(unittest.TestCase):
    """What `dist-status` reports is never finer than what the simulated final state shows (it may be coarser)."""

    def check(self, source: str):
        program = translate(parse(source, "v2"))
        reported = reported_groups(program)
        qubits, states, wires = final_state(program)
        qubits = {register: wires_of for register, wires_of in qubits.items() if register in reported}
        for group in true_groups(states, wires, qubits):
            if len(group) > 1:
                labels = {reported.get(register) for register in group}
                self.assertEqual(len(labels), 1, f"{source}: registers {sorted(group)} depend on each other but were reported in {labels}")
                self.assertNotIn(None, labels, f"{source}: registers {sorted(group)} depend on each other but were reported definite")

    def test_values_that_depend_on_each_other_are_reported_in_one_group(self):
        for source in ("a = 0..4; c = a + 1; return c;", "a = 0..4; p = 0..2; a += p; return a;",
                       "a = 0..4; b = a + 1; c = b - a; return c;",
                       "a = 0..4; x = 0; if a < 2 { x += 1; } return x;"):
            with self.subTest(source):
                self.check(source)

    def test_every_golden_program(self):
        for golden in GOLDEN:
            with self.subTest(golden.name):
                try:
                    self.check(golden.source)
                except AssertionError as error:
                    if "registers vary" in str(error):
                        self.skipTest(str(error))
                    raise


if __name__ == "__main__":
    unittest.main()

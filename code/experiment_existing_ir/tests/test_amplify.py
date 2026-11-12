import math
import subprocess
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, StatePreparation, ZGate
from qiskit.quantum_info import Statevector

from compiler.parser.parser import parse
from experiment_existing_ir import pipeline
from experiment_existing_ir.translate import TranslationError, translate

SHOTS = 8000


def qiskit_grover(n: int, marked: int, rounds: int):
    """Qiskit's own Grover operator on the equal superposition of 0 .. n - 1: the chance of each value."""
    qubits = max(1, math.ceil(math.log2(n)))
    amplitudes = np.zeros(2 ** qubits)
    amplitudes[:n] = 1 / math.sqrt(n)
    preparation = QuantumCircuit(qubits)
    preparation.append(StatePreparation(amplitudes), range(qubits))
    oracle = QuantumCircuit(qubits)
    zeros = [q for q in range(qubits) if not (marked >> q) & 1]
    for q in zeros:
        oracle.x(q)
    oracle.append(ZGate().control(qubits - 1) if qubits > 1 else ZGate(), range(qubits))
    for q in zeros:
        oracle.x(q)
    grover = GroverOperator(oracle, state_preparation=preparation)
    circuit = QuantumCircuit(qubits)
    circuit.compose(preparation, inplace=True)
    for _ in range(rounds):
        circuit.compose(grover, inplace=True)
    return Statevector(circuit).probabilities()[:n]


@unittest.skipUnless(pipeline.built(), "existing-ir-opt is not built: run helper/cpp/setup.sh and cpp/build.sh")
class Amplify(unittest.TestCase):
    def test_amplification_gives_the_chances_of_qiskit_s_grover_operator(self):
        # sizes that are and are not a power of two, and rounds past the best one, where the chance of the marked value falls again
        # (and ranges that start elsewhere, which only move the values: the reference is the same)
        for n, marked, rounds, start in ((4, 2, 2, 0), (8, 5, 3, 0), (16, 9, 2, 0), (6, 3, 2, 0), (5, 4, 2, 0), (8, 5, 2, 2), (8, 5, 1, -3)):
            with self.subTest(values=n, marked=marked, rounds=rounds, start=start):
                source = f"a = {start}..{start + n}; " + f"a ?= a == {marked + start}; " * rounds + "return a;"
                counts = pipeline.run(translate(parse(source, "v2")), shots=SHOTS)
                ours = np.array([counts.get(value + start, 0) / SHOTS for value in range(n)])
                self.assertLess(0.5 * np.abs(ours - qiskit_grover(n, marked, rounds)).sum(), 0.04)

    def test_what_cannot_be_amplified_yet_is_refused(self):
        for source in ("a = 3; a ?= a == 3; return a;",                              # not a distribution
                       "a = 0..4; k = 2; a ?= k == 2; return a;",                    # a predicate that does not depend on it
                       "a = 0..4; c = 0..2; if c == 1 { a ?= a == 2; } return a;"):  # inside a conditional
            with self.subTest(source), self.assertRaises(TranslationError):
                translate(parse(source, "v2"))
        for source in ("a = 0..4; p = 0..2; a += p; a ?= a == 2; return p;",  # a was changed by something else first, and p is tied to it
                       "a = 0..4; a += 1; a ?= a == 2; return a;"):            # a was changed by something else first
            with self.subTest(source), self.assertRaises(subprocess.CalledProcessError):
                pipeline.optimise(translate(parse(source, "v2")), *pipeline.DEFAULT)

    def test_a_function_name_stands_for_the_function_applied_to_the_variable(self):
        define = "function oracle(x) { return x + 3 == 5; }\n"
        self.assertEqual(translate(parse(define + "a = 0..8; a ?= oracle; return a;", "v2")),
                         translate(parse(define + "a = 0..8; a ?= oracle(a); return a;", "v2")))


if __name__ == "__main__":
    unittest.main()

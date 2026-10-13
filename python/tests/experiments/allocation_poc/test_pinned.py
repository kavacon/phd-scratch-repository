"""
Generated circuits must not change when the compiler's internals are reorganised: the OpenQASM text of every
preloaded program (at 6 qubits per register, 200 qubits) and its gate, depth and wire counts, before and after Qiskit's
optimiser. If one of these moves on purpose (a better adder, say), update the table and say why.
"""
import hashlib
import unittest

from compiler.passes.qasm import openqasm_text

from checking import qiskit_tools
from compiler.passes.optimise import optimise
from compiler.passes.qasm import openqasm_text
from compiler.system import System
from experiments.allocation_poc.experiment import compile_circuit
from experiments.allocation_poc.programs import PROGRAMS

PINNED = {  # program: (sha256 of the generated OpenQASM, generated [gates, two-qubit, depth, wires, touched], optimised [same])
    'area': ('18bb4daf21f4e5c2', [220, 213, 195, 30, 25], [3568, 1511, 2140, 30, 26]),
    'chain': ('4cf18825e0dd62ec', [92, 90, 70, 24, 19], [430, 220, 304, 24, 19]),
    'expression': ('99d69ddcffce407f', [158, 154, 125, 24, 19], [756, 384, 539, 24, 19]),
    'if-else': ('20341d1403bbc203', [150, 145, 136, 36, 26], [2669, 1146, 1704, 36, 28]),
    'one-armed': ('8244465db54e3768', [26, 24, 24, 24, 13], [489, 204, 335, 24, 15]),
    'nested': ('81bb722b7ddf9c1c', [76, 71, 65, 36, 20], [1104, 469, 679, 36, 21]),
    'loop': ('e7054e7fafa397a1', [276, 265, 185, 66, 61], [1754, 855, 1156, 66, 60]),
    'width': ('ff55d6e80e2b84eb', [237, 219, 173, 30, 25], [1478, 711, 938, 30, 26]),
    'explicit': ('55bc09b2cac53746', [94, 90, 72, 24, 19], [432, 220, 306, 24, 19]),
    'calls': ('2d137646e65a37ac', [74, 70, 47, 30, 25], [280, 148, 191, 30, 23]),
}


def key(measured):
    return [measured[k] for k in ("gates", "two_qubit", "depth", "wires", "touched")]


class PinnedCircuitsTestCase(unittest.TestCase):
    def test_every_program_generates_the_same_circuit_and_counts(self):
        for program in PROGRAMS:
            with self.subTest(program=program["id"]):
                digest, generated, optimised = PINNED[program["id"]]
                circuit = compile_circuit(program["source"], System(6, 200))
                self.assertEqual(digest, hashlib.sha256(openqasm_text(circuit).encode()).hexdigest()[:16])
                self.assertEqual(generated, key(qiskit_tools.measure(circuit)))
                self.assertEqual(optimised, key(qiskit_tools.measure(optimise(circuit))))


if __name__ == "__main__":
    unittest.main()

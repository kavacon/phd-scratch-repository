import itertools
import unittest

from checking.circuits import run_gates
from compiler.compiler import PassError
from compiler.passes.qasm import CircuitError, x_gates
from compiler.system import System
from experiments.encodings import experiment
from experiments.encodings.isolated import VARIANTS
from experiments.encodings.passes.rns import Layout, moduli_for, to_circuit_rns
from experiments.encodings.programs import PROGRAMS
from experiments.encodings.readout import Readout, decode, remainders_to_value

WIDTH = 4
SYSTEM = System(WIDTH, 100)
LAYOUT = Layout(moduli_for(WIDTH))
RANGE = 60  # 3 * 4 * 5
COMPARES = {"compare", "booleans", "in-place-branch", "if-else", "nested", "mixed"}  # the programs that order numbers


def bits_of(value):
    return [(value % m >> j) & 1 for m, size in zip(LAYOUT.moduli, LAYOUT.bits) for j in range(size)]


def value_in(state, slot):
    start, remainders = slot * LAYOUT.size, []
    for size in LAYOUT.bits:
        remainders.append(sum(state[start + j] << j for j in range(size)))
        start += size
    number = remainders_to_value(remainders, LAYOUT.moduli)
    return number - RANGE if number >= RANGE // 2 else number


class RnsTestCase(unittest.TestCase):
    def test_the_moduli_grow_with_the_width_and_the_readout_combines_the_remainders(self):
        self.assertEqual(((3, 4, 5), (3, 4, 5), (7, 8, 9), (7, 8, 9), (15, 16, 17)), tuple(moduli_for(w) for w in (3, 5, 6, 8, 9)))
        readout = Readout(1, "rns:3,4,5")
        for value in (-30, -5, 0, 11, 29):
            self.assertEqual(value, decode(readout, {sum(b << i for i, b in enumerate(bits_of(value))): 1}, WIDTH))

    def test_each_instruction_computes_its_result_and_restores_everything_else(self):
        for variant in VARIANTS:
            if variant.opcode in ("lt", "gt"):
                continue
            circuit = to_circuit_rns(variant.program(-3), SYSTEM)
            registers = variant.operands.count("r") + (1 if variant.update else 0)
            values = range(2) if variant.opcode in ("and", "or", "not") else range(-3, 4)
            for inputs in itertools.product(values, repeat=registers):
                state = [bit for x in inputs for bit in bits_of(x)]
                end = run_gates(x_gates(circuit), circuit.num_qubits, state + [0] * (circuit.num_qubits - len(state)))
                operands, taken = [], 0
                for kind in variant.operands:
                    operands.append(-3 if kind == "c" else inputs[int(variant.update) + taken])
                    taken += kind == "r"
                a, b = (operands + [0])[:2] if not variant.update else (inputs[0], operands[0])
                expected = {"copy": a, "add": a + b, "sub": a - b, "mul": a * b, "eq": int(a == b), "ne": int(a != b),
                            "and": int(bool(a & 1 and b & 1)), "or": int(bool((a | b) & 1)), "not": int(not a & 1)}[variant.opcode]
                slots = (circuit.num_qubits - 9) // LAYOUT.size  # the registers the instruction uses; the scratch qubits follow them
                self.assertEqual((expected + RANGE // 2) % RANGE - RANGE // 2, value_in(end, 0 if variant.update else slots - 1),
                                 (variant.id, inputs))
                self.assertFalse(any(end[circuit.num_qubits - 9:]), (variant.id, inputs))  # the scratch qubits are clean

    def test_programs_without_ordering_read_out_the_reference_and_the_others_are_refused(self):
        for program in PROGRAMS:
            if program["id"] in COMPARES:
                with self.assertRaises(CircuitError, msg=program["id"]) as caught:
                    try:
                        experiment.run(program["source"], ["rns"], SYSTEM, zx=False)
                    except PassError as error:
                        raise error.cause
                self.assertIn("ordering", str(caught.exception))
            else:
                self.assertTrue(all(r.agrees for r in experiment.run(program["source"], ["rns"], SYSTEM, zx=False)), program["id"])


if __name__ == '__main__':
    unittest.main()

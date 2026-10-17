import unittest

from qiskit.quantum_info import Statevector

from checking.simulate import simulate_isa
from demo.model import to_data
from experiments.encodings.demos import instructions
from experiments.encodings.encodings import ENCODINGS
from experiments.encodings.isolated import VARIANTS
from compiler.system import System

WIDTH = 3
CONSTANT = -2
SYSTEM = System(WIDTH, 100)
MASK = (1 << WIDTH) - 1


def signed(pattern):
    return pattern - (1 << WIDTH) if pattern >= 1 << (WIDTH - 1) else pattern


def semantics(opcode, a, b):
    """What an instruction leaves in its destination, from the operands' two's complement values."""
    return {
        "copy": a, "add": a + b, "sub": a - b, "mul": a * b, "lt": int(a < b), "gt": int(a > b), "eq": int(a == b),
        "ne": int(a != b), "and": int(bool(a & 1 and b & 1)), "or": int(bool((a | b) & 1)), "not": int(not a & 1),
    }[opcode] & MASK


def expected_registers(variant, values, slots):
    """The register patterns an instruction should leave, given the patterns it started from."""
    registers = list(values) + [0] * (slots - len(values))
    if variant.update:
        source = signed(CONSTANT & MASK) if variant.operands == "c" else signed(values[1])
        registers[0] = (values[0] + source if variant.opcode == "add" else values[0] - source) & MASK
        return registers
    operands, taken = [], 0
    for kind in variant.operands:
        operands.append(CONSTANT & MASK if kind == "c" else values[taken])
        taken += kind == "r"
    operands = [signed(o) for o in operands] + [0]
    registers[slots - 1] = semantics(variant.opcode, operands[0], operands[1])
    return registers


def pattern(registers):
    return sum(value << (slot * WIDTH) for slot, value in enumerate(registers))


class InstructionsTestCase(unittest.TestCase):
    def test_every_instruction_in_every_built_encoding_computes_its_result_and_restores_everything_else(self):
        import itertools
        for variant in VARIANTS:
            registers_in = variant.operands.count("r") + (1 if variant.update else 0)
            values = range(2) if variant.opcode in ("and", "or", "not") else range(1 << WIDTH)
            for name in ("binary", "binary-ttk", "fourier"):
                circuit = ENCODINGS[name].lowering(variant.program(CONSTANT, returns=True), SYSTEM).remove_final_measurements(inplace=False)
                slots = circuit.num_qubits // WIDTH - 1
                for inputs in itertools.product(values, repeat=registers_in):
                    final = Statevector.from_int(pattern(list(inputs) + [0] * (slots - registers_in)), 2 ** circuit.num_qubits).evolve(circuit)
                    state, probability = max(final.probabilities_dict().items(), key=lambda item: item[1])
                    self.assertEqual(pattern(expected_registers(variant, inputs, slots)), int(state, 2), (name, variant.id, inputs))
                    self.assertAlmostEqual(1.0, probability)

    def test_every_variant_is_a_well_formed_instruction_program(self):
        for variant in VARIANTS:
            simulate_isa(variant.program(5), SYSTEM)  # raises if it reads or writes a register it should not

    def test_the_page_shows_a_circuit_for_each_of_at_most_two_encodings(self):
        result = instructions.run(instructions.DEMO.clean_inputs({"case": "add-rr"}), ["binary", "fourier"])
        self.assertEqual(["note", "diagram", "note", "diagram"], [to_data(b)["type"] for b in result.stages[0].blocks])
        self.assertEqual(["binary", "binary-ttk"], instructions.DEMO.clean_steps(["fourier", "binary-ttk", "binary"]))


if __name__ == '__main__':
    unittest.main()

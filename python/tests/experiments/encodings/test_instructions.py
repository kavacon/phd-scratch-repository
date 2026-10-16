import itertools
import json
import unittest

from qiskit.quantum_info import Statevector

from checking.simulate import simulate_isa
from demo.model import to_data
from experiments.encodings.demos import instructions
from experiments.encodings.encodings import ENCODINGS
from experiments.encodings.isolated import VARIANTS
from compiler.parser.instructions import Const, Reg
from compiler.system import System

WIDTH = 3
BUILT = ("binary", "binary-ttk", "fourier")
CONSTANTS = (3, -2)


def signed(pattern, width=WIDTH):
    return pattern - (1 << width) if pattern >= 1 << (width - 1) else pattern


def semantics(opcode, values, width=WIDTH):
    """What an instruction leaves in its destination: a bit pattern, from the operands' two's complement values."""
    mask = (1 << width) - 1
    a, b = (values + [0])[:2]
    return {
        "copy": a, "add": a + b, "sub": a - b, "mul": a * b, "lt": int(a < b), "gt": int(a > b), "eq": int(a == b),
        "ne": int(a != b), "and": int(bool(a & 1 and b & 1)), "or": int(bool((a | b) & 1)), "not": int(not a & 1),
    }[opcode] & mask


def final_state(circuit, initial):
    """The circuit run on a basis state: the one basis state it ends in, and the probability of it."""
    unmeasured = circuit.remove_final_measurements(inplace=False)
    probabilities = Statevector.from_int(initial, 2 ** unmeasured.num_qubits).evolve(unmeasured).probabilities_dict()
    return max(probabilities.items(), key=lambda item: item[1])


def pattern_of(registers, width=WIDTH):
    return sum(value << (slot * width) for slot, value in enumerate(registers))


def cases(variant, constant):
    """Every assignment of operand values the instruction can see, as the patterns in its registers."""
    reg_sources = variant.operands.count("r")
    booleans = variant.opcode in ("and", "or", "not")
    values = range(2) if booleans else range(1 << WIDTH)
    return itertools.product(values, repeat=reg_sources + (1 if variant.update else 0))


class IsolatedProgramTestCase(unittest.TestCase):
    def test_every_variant_is_a_well_formed_instruction_program(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant.id):
                program = variant.program(5)
                simulate_isa(program, System(WIDTH, 100))  # raises if it reads or writes a register it should not

    def test_the_scratch_register_follows_the_registers_the_instruction_uses(self):
        program = VARIANTS[0].program(5, returns=True)  # ADD reg, reg
        self.assertEqual("R3", program.body[0].register)
        self.assertEqual(Reg("R2"), program.result)
        self.assertIsNone(VARIANTS[0].program(5).result)
        self.assertEqual("R1", next(v for v in VARIANTS if v.id == "update-add-c").program(5).body[0].register)

    def test_an_instruction_with_no_result_has_no_measurement_and_no_readout_conversion(self):
        add = VARIANTS[0]
        for name in BUILT:
            bare = ENCODINGS[name].lowering(add.program(5), System(WIDTH, 100))
            read = ENCODINGS[name].lowering(add.program(5, returns=True), System(WIDTH, 100))
            self.assertNotIn("measure", bare.count_ops(), name)
            self.assertIn("measure", read.count_ops(), name)
            self.assertLessEqual(bare.size(), read.size() - WIDTH, name)  # at least the measurements are left out
        fourier = ENCODINGS["fourier"].lowering(add.program(5), System(WIDTH, 100))
        read = ENCODINGS["fourier"].lowering(add.program(5, returns=True), System(WIDTH, 100))
        self.assertLess(fourier.size(), read.size() - WIDTH)  # and the inverse transform back to bits
        self.assertGreater(read.count_ops()["cp"], fourier.count_ops()["cp"])

    def test_a_constant_operand_is_the_one_given(self):
        program = next(v for v in VARIANTS if v.id == "add-rc").program(-7)
        self.assertEqual(Const(-7), program.body[-1].sources[1])

    def test_the_ids_are_unique(self):
        self.assertEqual(len(VARIANTS), len({v.id for v in VARIANTS}))


class InstructionCircuitTestCase(unittest.TestCase):
    """Each instruction alone, in each built encoding, on every operand value at width 3."""

    def check(self, name, variant, constant):
        system = System(WIDTH, 100)
        circuit = ENCODINGS[name].lowering(variant.program(constant, returns=True), system)  # read out, so it ends in a basis state
        slots = circuit.num_qubits // WIDTH - 1
        mask = (1 << WIDTH) - 1
        for inputs in cases(variant, constant):
            values = list(inputs)
            registers = [0] * slots
            if variant.update:
                registers[0] = values[0]
                source_values = [constant & mask] if variant.operands == "c" else [values[1]]
                registers[1:1 + len(values) - 1] = values[1:]
                dst = registers[0]
                other = signed(source_values[0])
                registers[0] = (dst + other if variant.opcode == "add" else dst - other) & mask
                initial = pattern_of([values[0], *values[1:]] + [0] * (slots - len(values)))
                expected = pattern_of([registers[0], *values[1:]] + [0] * (slots - len(values)))
            else:
                operands, k = [], 0
                for kind in variant.operands:
                    if kind == "c":
                        operands.append(constant & mask)
                    else:
                        operands.append(values[k]); k += 1
                regs = values + [0] * (slots - len(values))
                initial = pattern_of(regs)
                regs[slots - 1] = semantics(variant.opcode, [signed(o) for o in operands])
                expected = pattern_of(regs)
            state, probability = final_state(circuit, initial)
            got = int(state, 2)  # qubit 0 is the rightmost character
            self.assertEqual(expected, got, (name, variant.id, constant, inputs))
            self.assertAlmostEqual(1.0, probability, msg=(name, variant.id, constant, inputs))

    def test_every_instruction_in_every_encoding_computes_its_result_and_restores_everything_else(self):
        for variant in VARIANTS:
            constants = CONSTANTS if "c" in variant.operands else (0,)
            for name in BUILT:
                for constant in constants:
                    with self.subTest(encoding=name, variant=variant.id, constant=constant):
                        self.check(name, variant, constant)


class InstructionsDemoTestCase(unittest.TestCase):
    def run_case(self, case, enabled=("binary", "fourier"), **inputs):
        return instructions.run(instructions.DEMO.clean_inputs({"case": case, **inputs}), list(enabled))

    def test_every_instruction_shows_just_the_circuits(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant.id):
                result = self.run_case(variant.id)
                self.assertIsNone(result.error)
                self.assertEqual(["circuits"], [s.id for s in result.stages])
                kinds = [to_data(b)["type"] for b in result.stages[0].blocks]
                self.assertEqual(["note", "diagram", "note", "diagram"], kinds)  # a heading and a circuit for each encoding
                json.dumps(to_data(result))

    def test_the_circuit_is_the_one_the_lowering_builds_with_no_optimiser(self):
        result = self.run_case("add-rr", enabled=["binary"])
        picture = result.stages[0].blocks[1].picture
        lowered = ENCODINGS["binary"].lowering(VARIANTS[0].program(5), System(4, 100))
        self.assertEqual(lowered.size(), picture["total"])  # every gate and measurement of the lowering's own circuit

    def test_the_instruction_and_what_it_does_head_the_page(self):
        stage = self.run_case("add-rc", constant=9).stages[0]
        self.assertEqual("ADD reg, const", stage.name)
        self.assertIn("ADD", stage.summary)
        self.assertIn("9", stage.summary)

    def test_an_encoding_that_is_not_built_yet_is_reported_under_its_heading(self):
        blocks = self.run_case("add-rr", enabled=["binary", "angle"]).stages[0].blocks
        self.assertEqual(["note", "diagram", "note", "note"], [to_data(b)["type"] for b in blocks])
        self.assertEqual("bad", blocks[3].tone)
        self.assertIn("NotImplementedError", blocks[3].text)

    def test_the_selector_comes_first_and_allows_two_encodings_with_binary_and_fourier_on(self):
        demo = instructions.DEMO
        boot = demo.bootstrap()
        self.assertTrue(boot["steps_first"])
        self.assertEqual(2, boot["max_steps"])
        self.assertEqual(["binary", "fourier"], [s.id for s in demo.steps if s.enabled])
        self.assertEqual(["binary", "binary-ttk"], demo.clean_steps(["fourier", "binary-ttk", "binary"]))  # never more than two

    def test_nothing_chosen_says_so_and_the_constant_is_kept_in_range(self):
        self.assertEqual(1, len(self.run_case("add-rr", enabled=[]).stages[0].blocks))
        cleaned = [instructions.DEMO.clean_inputs({"constant": c})["constant"] for c in (999, -999)]
        self.assertEqual([127, -128], cleaned)


if __name__ == '__main__':
    unittest.main()

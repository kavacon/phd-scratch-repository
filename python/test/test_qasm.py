import itertools
import unittest

from allocation_poc.circuits import gates_from_text, leftover_qubits, run_gates, value_of
from allocation_poc.programs import PROGRAMS
from allocation_poc.simulate import simulate_isa
from parser.parser import parse
from passes.allocate import allocate
from passes.isa import lower_to_isa
from parser.instructions import Const
from passes.qasm import C, CircuitError, Gate, Q, _Builder, gate_text, to_openqasm

W = 3
MASK = (1 << W) - 1
SCRATCH = 3 * W  # registers a, b and the destination come first, then the scratch register
QUBITS = 4 * W

A = [Q(i) for i in range(W)]
B = [Q(W + i) for i in range(W)]
D = [2 * W + i for i in range(W)]
CONSTANT = lambda value: [C((value >> i) & 1) for i in range(W)]

EXPECTED = {
    "copy": lambda a, b: a, "add": lambda a, b: (a + b) & MASK, "sub": lambda a, b: (a - b) & MASK,
    "mul": lambda a, b: (a * b) & MASK, "lt": lambda a, b: int(a < b), "gt": lambda a, b: int(a > b),
    "eq": lambda a, b: int(a == b), "ne": lambda a, b: int(a != b),
}


def initial(a, b):
    return [(a >> i) & 1 for i in range(W)] + [(b >> i) & 1 for i in range(W)]


def build(opcode, *srcs):
    builder = _Builder(W, SCRATCH)
    builder.op(opcode, D, list(srcs))
    return builder.gates


def decode(state, start):
    return sum(state[start + i] << i for i in range(W))


class OpcodeCircuitTestCase(unittest.TestCase):
    def check(self, opcode, operands, a, b, expected):
        gates = build(opcode, *operands)
        state = run_gates(gates, QUBITS, initial(a, b))
        label = f"{opcode} a={a} b={b}"
        self.assertEqual(expected, decode(state, 2 * W), label)
        self.assertEqual((a, b), (decode(state, 0), decode(state, W)), f"{label}: operands changed")
        self.assertEqual(0, decode(state, 3 * W), f"{label}: scratch left dirty")
        back = run_gates(list(reversed(gates)), QUBITS, state)  # the inverse instruction is the reversed gates
        self.assertEqual(0, decode(back, 2 * W), f"{label}: inverse did not clear the destination")
        self.assertEqual(initial(a, b), back[:2 * W], f"{label}: inverse changed an operand")

    def test_every_opcode_on_two_registers_for_every_input(self):
        for opcode, f in EXPECTED.items():
            if opcode == "copy":
                continue
            for a, b in itertools.product(range(1 << W), repeat=2):
                self.check(opcode, (A, B), a, b, f(a, b))

    def test_copy_on_a_register(self):
        for a in range(1 << W):
            self.check("copy", (A,), a, 0, a)

    def test_every_opcode_with_a_constant_operand_on_either_side(self):
        for opcode, f in EXPECTED.items():
            if opcode == "copy":
                continue
            for a, c in itertools.product(range(1 << W), repeat=2):
                self.check(opcode, (A, CONSTANT(c)), a, 0, f(a, c))
                self.check(opcode, (CONSTANT(c), A), a, 0, f(c, a))

    def test_an_operation_on_two_constants_is_computed_at_compile_time(self):
        for opcode, f in EXPECTED.items():
            if opcode == "copy":
                continue
            gates = build(opcode, CONSTANT(5), CONSTANT(3))
            self.assertTrue(all(not g.controls for g in gates), opcode)  # only X gates
            self.assertEqual(f(5, 3), decode(run_gates(gates, QUBITS), 2 * W))

    def test_booleans_live_in_bit_zero(self):
        for a, b in itertools.product((0, 1), repeat=2):
            self.check("and", (A, B), a, b, a & b)
            self.check("or", (A, B), a, b, a | b)
            self.check("not", (A,), a, 0, 1 - a)

    def test_the_same_register_on_both_sides(self):
        for a in range(1 << W):
            self.check("add", (A, A), a, 0, (2 * a) & MASK)
            self.check("sub", (A, A), a, 0, 0)
            self.check("lt", (A, A), a, 0, 0)
            self.check("gt", (A, A), a, 0, 0)
            self.check("eq", (A, A), a, 0, 1)
            self.check("ne", (A, A), a, 0, 0)

    def test_squaring_and_division_are_refused(self):
        with self.assertRaises(CircuitError):
            build("mul", A, A)
        with self.assertRaises(CircuitError):
            build("div", A, B)
        with self.assertRaises(CircuitError):
            build("div", CONSTANT(4), CONSTANT(2))

    def test_the_circuit_only_touches_the_scratch_qubit_it_needs(self):
        used = {g.target for g in build("add", A, B)} | {q for g in build("add", A, B) for q, _ in g.controls}
        self.assertEqual({SCRATCH}, used & set(range(SCRATCH, QUBITS)))  # one carry qubit, not the whole register
        self.assertFalse({g.target for g in build("copy", A)} & set(range(SCRATCH, QUBITS)))


class ControlledRegionTestCase(unittest.TestCase):
    def circuit(self, source, width=6):
        return to_openqasm(allocate(lower_to_isa(parse(source)), width, 1000))

    def test_the_two_regions_of_an_if_are_controlled_on_one_qubit_with_opposite_polarity(self):
        circuit = self.circuit("a = 3; c = a < 5; y = 0; if c { y = a + 1; } else { y = a - 1; } return y;")
        polarities = {}
        for gate in circuit.gates:
            for qubit, polarity in gate.controls:
                polarities.setdefault(qubit, set()).add(polarity)
        self.assertTrue(any(seen == {True, False} for seen in polarities.values()))  # the condition bit, tested both ways

    def test_the_text_holds_gates_and_measurements_only(self):
        circuit = self.circuit("a = 3; b = a + 2; return b;")
        for line in circuit.text.splitlines():
            if line and not line.startswith(("OPENQASM", "include", "qubit", "bit")):
                self.assertTrue(line.startswith(("x ", "cx ", "ccx ", "ctrl", "negctrl", "result[")), line)
        self.assertNotIn("ADD", circuit.text)  # no instruction text

    def test_the_circuit_follows_whichever_branch_the_condition_selects(self):
        for a, expected in ((3, 4), (9, 8)):
            circuit = self.circuit(f"a = {a}; c = a < 5; y = 0; if c {{ y = a + 1; }} else {{ y = a - 1; }} return y;")
            state = run_gates(circuit.gates, circuit.qubits)
            self.assertEqual(expected, value_of(state, circuit.result))
            self.assertEqual([], leftover_qubits(state, circuit.result))


class ProgramCircuitTestCase(unittest.TestCase):
    EXTRA = [
        "x = 5; y = 3; z = 0; if x > y { z = x - y; } else { z = y - x; } return z;",
        "a = 7; b = 2; c = a != b; d = (a == 7) && c; return d;",
        "x = 6; return !(x < 3) || (x == 2);",
    ]

    def sources(self):
        return [p["source"] for p in PROGRAMS] + self.EXTRA

    def test_every_program_gives_its_result_with_nothing_left_over_at_several_widths(self):
        for width in (6, 8):
            for source in self.sources():
                with self.subTest(width=width, source=source[:40]):
                    isa = lower_to_isa(parse(source))
                    circuit = to_openqasm(allocate(isa, width, 1000))
                    state = run_gates(circuit.gates, circuit.qubits)
                    self.assertEqual(int(simulate_isa(isa).value) & ((1 << width) - 1), value_of(state, circuit.result))
                    self.assertEqual([], leftover_qubits(state, circuit.result))

    def test_the_printed_text_reads_back_as_the_same_gates(self):
        for source in self.sources():
            circuit = to_openqasm(allocate(lower_to_isa(parse(source)), 6, 1000))
            qubits, gates = gates_from_text(circuit.text)
            self.assertEqual(circuit.qubits, qubits)
            normal = lambda gs: [(g.target, sorted(g.controls)) for g in gs]  # the order of controls does not matter
            self.assertEqual(normal(circuit.gates), normal(gates))

    def test_the_layout_is_one_register_with_the_scratch_after_the_slots(self):
        allocation = allocate(lower_to_isa(parse(PROGRAMS[0]["source"])), 4, 100)
        circuit = to_openqasm(allocation)
        self.assertEqual((allocation.needed + 1) * 4, circuit.qubits)
        self.assertIn(f"qubit[{circuit.qubits}] q;", circuit.text)
        self.assertTrue(all(g.target < circuit.qubits for g in circuit.gates))

    def test_a_division_program_is_refused(self):
        with self.assertRaises(CircuitError):
            to_openqasm(allocate(lower_to_isa(parse("a = 8; b = a / 2; return b;")), 4, 100))


class MeasurementTestCase(unittest.TestCase):
    def circuit(self, source, width=4):
        return to_openqasm(allocate(lower_to_isa(parse(source)), width, 1000))

    def test_the_returned_register_is_measured_into_a_classical_register_and_nothing_else_is(self):
        circuit = self.circuit("a = 3; b = a + 2; return b;")
        lines = circuit.text.splitlines()
        self.assertIn("bit[4] result;", lines)
        measures = [line for line in lines if "measure" in line and not line.startswith("//")]
        self.assertEqual(4, len(measures))
        self.assertEqual([f"result[{i}] = measure q[{q}];" for i, q in enumerate(circuit.measured)], measures)
        self.assertEqual([b.index for b in circuit.result], list(circuit.measured))

    def test_the_measurements_come_after_every_gate(self):
        lines = [l for l in self.circuit("a = 3; b = a + 2; return b;").text.splitlines() if l and not l.startswith("//")]
        first = next(i for i, l in enumerate(lines) if "measure" in l)
        self.assertTrue(all("measure" in l for l in lines[first:]))

    def test_a_program_that_returns_a_literal_returns_a_register_holding_it_like_any_other(self):
        circuit = self.circuit("return 7;")
        self.assertEqual(4, circuit.text.count("= measure q["))
        state = run_gates(circuit.gates, circuit.qubits)
        self.assertEqual(7, value_of(state, circuit.result))
        self.assertEqual([], leftover_qubits(state, circuit.result))

    def test_a_program_that_does_not_return_a_register_is_refused(self):
        from parser.instructions import IsaProgram
        allocation = allocate(lower_to_isa(parse("return 7;")), 4, 1000)
        from dataclasses import replace
        broken = replace(allocation, program=IsaProgram(allocation.program.body, Const(7)))
        with self.assertRaises(CircuitError):
            to_openqasm(broken)

    def test_a_constant_that_is_computed_first_still_comes_back(self):
        circuit = self.circuit("a = 3; b = a + 2; return 5;")
        state = run_gates(circuit.gates, circuit.qubits)
        self.assertEqual(5, value_of(state, circuit.result))
        self.assertEqual([], leftover_qubits(state, circuit.result))

    def test_measuring_does_not_change_the_gates(self):
        for source in ("a = 3; b = a + 2; return b;", "a = 5; return a;"):
            circuit = self.circuit(source)
            self.assertEqual(len(circuit.gates), len(gates_from_text(circuit.text)[1]))


class GateTextTestCase(unittest.TestCase):
    def test_gates_are_written_with_the_short_names_where_there_is_one(self):
        self.assertEqual("x q[2];", gate_text(Gate(2)))
        self.assertEqual("cx q[0], q[2];", gate_text(Gate(2, ((0, True),))))
        self.assertEqual("ccx q[0], q[1], q[2];", gate_text(Gate(2, ((0, True), (1, True)))))
        self.assertEqual("ctrl(3) @ x q[0], q[1], q[2], q[3];", gate_text(Gate(3, ((0, True), (1, True), (2, True)))))
        self.assertEqual("negctrl @ x q[0], q[1];", gate_text(Gate(1, ((0, False),))))
        self.assertEqual("ctrl @ negctrl(2) @ x q[0], q[1], q[2], q[3];", gate_text(Gate(3, ((1, False), (0, True), (2, False)))))


if __name__ == '__main__':
    unittest.main()

import unittest

from checking.circuits import register_probabilities
from checking.simulate import simulate_isa
from experiments.distributions import experiment
from experiments.distributions.programs import PROGRAMS

EXTRA = [
    "a = 2..6;\nreturn a;\n",  # a power of two that does not start at a multiple of it
    "a = -2..2;\nreturn a;\n",  # a signed range
    "a = 0..3;\nb = 0;\nc = 0..2;\nif c {\n    b = a + 1;\n}\nreturn b;\n",  # a conditional on a distribution reading another
    "a = 0..4;\nx = 1;\nif a {\n    x = 5;\n}\nreturn x;\n",  # a condition that is 2 or 3 is true, as the machine says
]


MOST_QUBITS = 16  # the state vector holds every qubit, so the largest circuits are left to the run on the simulator


def circuit_and_state(source):
    compilation = experiment.compiler().run(source, until="to_openqasm")
    if compilation.circuit.num_qubits > MOST_QUBITS:
        return None, None
    names = [register.name for register in compilation.circuit.qregs]
    held, state = simulate_isa(compilation["allocate"], experiment.SYSTEM).state
    expected = {}
    for key, amplitude in state.items():
        values = dict(zip(held, key, strict=True))
        full = tuple(values.get(name, 0) for name in names)  # a register not held, and scratch, are zero
        expected[full] = expected.get(full, 0.0) + abs(amplitude) ** 2
    return register_probabilities(compilation.circuit, experiment.SYSTEM.register_width), expected


class CircuitTestCase(unittest.TestCase):
    def test_the_circuit_ends_in_the_state_the_instruction_machine_computes(self):
        for source in [p["source"] for p in PROGRAMS] + EXTRA:
            with self.subTest(source=source):
                built, expected = circuit_and_state(source)
                if built is None:
                    continue
                self.assertEqual(set(expected), set(built))
                for key, p in expected.items():
                    self.assertAlmostEqual(p, built[key], places=9)


if __name__ == '__main__':
    unittest.main()

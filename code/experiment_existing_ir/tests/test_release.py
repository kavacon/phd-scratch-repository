import re
import subprocess
import unittest

from compiler.parser.parser import parse
from compiler.system import System
from experiment_existing_ir import metrics, pipeline
from experiment_existing_ir.translate import translate
from tests.golden import GOLDEN, TOLERANCE, distance, entry

# Programs that only fit a small system because dead registers are released and their qubits reused: the qubits each needs without
# release are more than the budget here. The intended outcome does not depend on the budget.
TIGHT = {"a function called in a loop": 20, "a loop variable compared with a distribution": 27,
         "every comparison of a distribution": 24, "a condition on a distribution in a loop": 19,
         "two variables tied to one condition": 29, "boolean operators": 17,
         "a distribution conditional inside another, in a loop": 24}
# These give the wrong outcome when a register is freed without being returned to zero first, because the qubits are reused.
SENSITIVE = tuple(TIGHT)


def compile_with(name: str, qubits=None, strategy="lazy", uncompute=True, policy="spread") -> str:
    program = translate(parse(entry(name).source, "v2"))
    if qubits is not None:
        program = pipeline.with_system(program, System(qubits=qubits))
    passes = tuple(p.replace("strategy=lazy", f"strategy={strategy}").replace("--release-registers", f"--release-registers{{uncompute={str(uncompute).lower()}}}")
                   .replace("--assign-qubits", f"--assign-qubits{{policy={policy}}}")
                   for p in pipeline.DEFAULT)
    return pipeline.optimise(program, *passes)


def pool(qc: str) -> int:
    return int(re.search(r"pool_qubits = (\d+)", qc).group(1))


def outcome(name: str, qc: str) -> float:
    return distance(pipeline.execute(pipeline.export(qc, "openqasm3"), "openqasm3", "aer", 3000), entry(name).outcomes)


@unittest.skipUnless(pipeline.built(), "existing-ir-opt is not built: run helper/cpp/setup.sh and cpp/build.sh")
class Release(unittest.TestCase):
    def test_a_program_fits_the_qubits_release_gives_it_and_still_gives_the_intended_outcome(self):
        for name, qubits in TIGHT.items():
            with self.subTest(name):
                self.assertGreater(pool(compile_with(name)), qubits)  # without release it needs more than the system has
                tight = compile_with(name, qubits)
                self.assertLessEqual(pool(tight), qubits)
                self.assertLess(outcome(name, tight), TOLERANCE)

    def test_a_program_that_does_not_fit_even_with_release_is_refused_and_a_system_that_is_large_enough_releases_nothing(self):
        name = "a loop variable compared with a distribution"
        with self.assertRaises(subprocess.CalledProcessError) as raised:
            compile_with(name, 10)
        self.assertIn("even releasing every register that may be released", raised.exception.stderr)
        self.assertIn("released_registers = 0", compile_with(name, 100))  # lazy: nothing needs releasing

    def test_releasing_eagerly_gives_the_intended_outcome_for_every_program(self):
        for golden in GOLDEN:
            with self.subTest(golden.name):
                self.assertLess(outcome(golden.name, compile_with(golden.name, strategy="eager")), TOLERANCE)

    def test_the_outcome_depends_on_returning_a_register_to_zero_before_its_qubits_are_reused(self):
        for name in SENSITIVE:
            with self.subTest(name):
                broken = compile_with(name, TIGHT[name], uncompute=False, policy="lowest")  # reuse as hard as possible
                self.assertGreater(outcome(name, broken), 0.2)  # the same qubits, reused dirty: not the intended outcome

    def test_given_more_qubits_than_it_needs_a_program_uses_them_to_wait_less_and_never_more_than_it_is_given(self):
        name = "a loop variable compared with a distribution"
        needed = pool(compile_with(name))
        tight, roomy = compile_with(name, needed), compile_with(name, 2 * needed)
        self.assertEqual(pool(tight), needed)  # exactly enough: every qubit is shared as it must be
        self.assertLessEqual(pool(roomy), 2 * needed)
        self.assertGreater(pool(roomy), needed)  # the spare qubits are used
        self.assertLess(metrics.of_qc(roomy).depth, metrics.of_qc(tight).depth)  # and so less waits for a qubit that is being cleared
        self.assertLess(outcome(name, roomy), TOLERANCE)


if __name__ == "__main__":
    unittest.main()

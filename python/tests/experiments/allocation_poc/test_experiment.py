import unittest

from checking.simulate import execute
from compiler.compiler import PassError
from compiler.parser.instructions import Alloc, Scratch
from compiler.passes.allocate import WidthExceeded
from compiler.system import System
from experiments.allocation_poc.experiment import FRONT_END, compiler, text_of
from experiments.allocation_poc.programs import PROGRAMS

IDS = compiler().names
KEPT = ("unroll", "flatten", "balance", "rename", "lower_allocation")  # up to the releases, with functions kept as scopes


def source_of(program_id):
    return next(p for p in PROGRAMS if p["id"] == program_id)["source"]


def stages(source, ids=None, width=6, qubits=100):
    """What each chosen pass produced, by name. By default every pass up to the circuit (not Qiskit's optimiser)."""
    run = compiler(System(width, qubits))
    skip = ["optimise"] + ([n for n in run.names if n not in ids] if ids is not None else [])
    return {stage.name: stage.output for stage in run.run(source, skip=skip).stages}


def value(output):
    return execute(output).value


def leaks(output):
    return [name for names in execute(output).leaks.values() for name in names]


class PipelineTestCase(unittest.TestCase):
    def test_the_ast_passes_come_first_in_the_order_the_compile_function_runs_them(self):
        self.assertEqual(["inline", "unroll", "flatten", "balance", "rename", "lower_allocation",
                          "lower_to_isa", "allocate", "to_openqasm", "optimise"], IDS)
        self.assertEqual(IDS[:len(FRONT_END)], [fn.compile_pass.name for fn in FRONT_END])


class InlineTestCase(unittest.TestCase):
    def test_the_inline_stage_removes_functions_and_keeps_the_result(self):
        for program in PROGRAMS:
            run = stages(program["source"])
            self.assertNotIn("function", text_of(run["inline"]), program["id"])
            self.assertEqual(value(run["source"]), value(run["inline"]), program["id"])


class AllocationTestCase(unittest.TestCase):
    def allocation(self, program_id, width, qubits):
        return stages(source_of(program_id), None, width, qubits)["allocate"]

    def test_the_allocation_declares_the_scratch_register_after_the_slots(self):
        program = self.allocation("if-else", width=2, qubits=14)
        slots = {i.register for i in program.body if isinstance(i, Alloc)}
        self.assertEqual(5, len(slots))
        self.assertEqual(["R5"], [i.register for i in program.body if isinstance(i, Scratch)])  # after R0..R4
        self.assertIn("R0", text_of(program))
        self.assertEqual([], leaks(program))

    def test_the_result_is_unchanged_by_allocation_for_every_program(self):
        for program in PROGRAMS:
            run = stages(program["source"])
            self.assertEqual(value(run["lower_to_isa"]), value(run["allocate"]), program["id"])

    def test_too_few_qubits_raises_with_the_packing_it_would_have_needed(self):
        with self.assertRaises(PassError) as caught:
            self.allocation("if-else", width=4, qubits=16)
        self.assertEqual("allocate", caught.exception.name)
        self.assertIsInstance(caught.exception.cause, WidthExceeded)
        self.assertIn("needs 5 register slots but only 3 are available", str(caught.exception.cause))
        self.assertIn("SCRATCH R5", text_of(caught.exception.partial))  # the packing it would have needed, 5 slots


class PassesTestCase(unittest.TestCase):

    def test_every_preloaded_program_keeps_its_result_and_ends_without_leftovers(self):
        for program in PROGRAMS:
            with self.subTest(program=program["id"]):
                run = stages(program["source"], KEPT)
                self.assertEqual(["source", *KEPT], list(run))
                self.assertEqual(1, len({value(output) for output in run.values()}))
                self.assertEqual([], leaks(run["lower_allocation"]))

    def test_early_release_lowers_peak_width(self):
        run = stages(source_of("width"), KEPT)
        peak = lambda output: max(execute(output).peak.values())  # noqa: E731
        self.assertLess(peak(run["lower_allocation"]), peak(run["rename"]))


if __name__ == '__main__':
    unittest.main()

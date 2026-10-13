import unittest

from checking.simulate import SimulationError, execute, simulate
from compiler.compiler import PassError
from compiler.parser.instructions import Alloc, Scratch
from compiler.parser.parser import parse
from compiler.passes.allocate import WidthExceeded
from compiler.passes.isa import IsaError
from compiler.system import System
from demo.text_diff import diff_lines
from compiler.passes.qasm import openqasm_text
from experiments.allocation_poc.experiment import FRONT_END, compile_circuit, compiler, text_of
from experiments.allocation_poc.passes.rename import RenameError
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

    def test_the_compilation_starts_with_the_source_and_has_one_stage_per_pass(self):
        self.assertEqual(["source", "unroll", "flatten"], list(stages(PROGRAMS[0]["source"], ["unroll", "flatten"])))

    def test_every_stage_has_text(self):
        for name, output in stages(PROGRAMS[0]["source"]).items():
            self.assertTrue(text_of(output).strip(), name)

    def test_compile_circuit_is_the_generated_circuit(self):
        self.assertEqual(text_of(stages(PROGRAMS[0]["source"])["to_openqasm"]), openqasm_text(compile_circuit(PROGRAMS[0]["source"])))

    def test_a_stage_that_is_skipped_leaves_the_next_pass_to_say_what_it_expected(self):
        with self.assertRaises(PassError) as caught:
            compiler().run(PROGRAMS[0]["source"], skip=["lower_to_isa"])
        self.assertEqual("allocate", caught.exception.name)
        self.assertIn("skipped", str(caught.exception))
        self.assertEqual("lower_allocation", caught.exception.stages[-1].name)


class InlineTestCase(unittest.TestCase):
    def test_the_inline_stage_removes_functions_and_keeps_the_result(self):
        for program in PROGRAMS:
            run = stages(program["source"])
            self.assertNotIn("function", text_of(run["inline"]), program["id"])
            self.assertEqual(value(run["source"]), value(run["inline"]), program["id"])

    def test_without_inline_the_instruction_pass_says_functions_must_be_inlined(self):
        with self.assertRaises(PassError) as caught:
            stages(PROGRAMS[0]["source"], [i for i in IDS if i != "inline"])
        self.assertEqual("lower_to_isa", caught.exception.name)
        self.assertIsInstance(caught.exception.cause, IsaError)
        self.assertIn("inlined", str(caught.exception))


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
    def test_the_instruction_stage_is_printed_as_instructions_and_run_on_the_instruction_machine(self):
        for program in PROGRAMS:
            run = stages(program["source"], [i for i in IDS if i in ("inline", *KEPT, "lower_to_isa")])
            self.assertIn("ALLOC", text_of(run["lower_to_isa"]))
            self.assertEqual(value(run["lower_allocation"]), value(run["lower_to_isa"]), program["id"])
            self.assertEqual([], leaks(run["lower_to_isa"]), program["id"])

    def test_each_pass_changes_what_it_says_it_does(self):
        run = stages(PROGRAMS[0]["source"], KEPT)
        names = list(run)

        def changes(step):
            before = text_of(run[names[names.index(step) - 1]]).splitlines()
            _, after = diff_lines(before, text_of(run[step]).splitlines())
            return sum(e["kind"] == "changed" for e in after), sum(e["kind"] == "added" for e in after)

        self.assertEqual((1, 1), changes("flatten"))  # the call is returned through a temporary
        self.assertEqual(8, changes("rename")[0])
        self.assertEqual(3, sum("~=" in line for line in text_of(run["lower_allocation"]).splitlines()))  # p in area, then y and x in the caller

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

    def test_a_pass_run_without_what_it_expects_fails_and_says_which_pass(self):
        source = "function f(a: int): int { if a < 2 { a = 1; } return a; } return f(1);"
        with self.assertRaises(PassError) as caught:
            stages(source, ["rename", "lower_allocation"])
        self.assertIsInstance(caught.exception.cause, RenameError)
        self.assertIn("balance", str(caught.exception))

    def test_programs_that_cannot_run_say_why(self):
        with self.assertRaises(SimulationError) as caught:
            simulate(parse("function f(n: int): int { return g(n); } return f(1);"))
        self.assertIn("not defined", str(caught.exception))
        endless = "function f(n: int): int { x = n; for i in 0 .. 10000000 { x = x + 1; } return x; } return f(1);"
        with self.assertRaises(SimulationError) as caught:
            simulate(parse(endless))
        self.assertIn("step limit", str(caught.exception))


if __name__ == '__main__':
    unittest.main()

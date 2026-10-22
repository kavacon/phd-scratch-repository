import unittest

from checking.simulate import simulate_isa
from compiler.parser.instructions import Ctrl, EndCtrl, ICtrl
from experiments.distributions import experiment
from experiments.distributions.programs import PROGRAMS

SOURCES = {p["id"]: p["source"] for p in PROGRAMS}


def compile_to_release(source, flatten=True):
    skip = ["flatten_conditionals"] + ([] if flatten else ["flatten_controls"])  # the source is left nested to test the instructions
    return experiment.compiler().run(source, until="release_values", skip=skip).output


def depth(program):
    now = deepest = 0
    for instruction in program.body:
        now += isinstance(instruction, (Ctrl, ICtrl))
        now -= isinstance(instruction, EndCtrl)
        deepest = max(deepest, now)
    return deepest


class FlattenControlsTestCase(unittest.TestCase):
    def test_nested_regions_become_regions_on_one_register_with_the_same_result_and_registers_held(self):
        source = SOURCES["branch-nested"]
        nested, flat = compile_to_release(source, flatten=False), compile_to_release(source)
        self.assertEqual((2, 1), (depth(nested), depth(flat)))
        held_nested, held_flat = (simulate_isa(p, experiment.SYSTEM) for p in (nested, flat))
        self.assertEqual(held_nested.value, held_flat.value)
        self.assertEqual(set(held_nested.state[0]), set(held_flat.state[0]))  # the conjunctions are computed and undone

    def test_a_program_with_no_nested_region_is_left_alone(self):
        source = SOURCES["branch-copy"]
        self.assertEqual(compile_to_release(source, flatten=False), compile_to_release(source))


if __name__ == '__main__':
    unittest.main()

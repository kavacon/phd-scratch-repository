import unittest

from compiler.parser.parser import parse
from experiment_existing_ir import pipeline
from experiment_existing_ir.pipeline import Export
from experiment_existing_ir.translate import translate
from tests.golden import GOLDEN, TOLERANCE, distance, entry

SHOTS = 4000
BUILT = pipeline.built()
# (format, simulator): OpenQASM 3 on Aer (Qiskit's own importer) and on DDSIM; QIR on DDSIM; Jeff read back by MQT.
READERS = (("openqasm3", "aer"), ("openqasm3", "ddsim"), ("qir-base", "ddsim"), ("qir-adaptive", "ddsim"), ("jeff", "mqt"))


def outcome(name: str, format: str = "openqasm3", backend: str = "aer") -> float:
    """How far the program's sampled outcomes are from the golden entry, when the compiled artifact is run."""
    golden = entry(name)
    artifact = pipeline.build(translate(parse(golden.source, "v2")), *pipeline.DEFAULT, Export(format))
    return distance(pipeline.execute(artifact, format, backend, SHOTS), golden.outcomes)


@unittest.skipUnless(BUILT, "existing-ir-opt is not built: run helper/cpp/setup.sh and cpp/build.sh")
class Golden(unittest.TestCase):
    """Programs run through the pipeline give the outcome the language intends."""

    def test_products_comparisons_and_booleans_on_integers_and_distributions(self):
        for name in ("a product of negative integers", "a product of two distributions", "boolean operators",
                     "equality of two distributions"):
            self.assertLess(outcome(name), TOLERANCE, name)

    def test_an_interval_may_be_any_size_and_start_anywhere(self):
        for name in ("an interval that is not a power of two", "an interval that does not start at zero"):
            self.assertLess(outcome(name), TOLERANCE, name)

    def test_every_entry(self):
        for golden in GOLDEN:
            with self.subTest(golden.name):
                self.assertLess(outcome(golden.name), TOLERANCE)


@unittest.skipUnless(BUILT, "existing-ir-opt is not built: run helper/cpp/setup.sh and cpp/build.sh")
class Formats(unittest.TestCase):
    """An export format is a pipeline stage; the artifact it gives, run on a simulator that did not produce it, gives the
    outcome the language intends."""

    def test_every_format_gives_the_intended_outcome_on_every_simulator_that_reads_it(self):
        for format, backend in READERS:
            self.assertLess(outcome("conditional on a distribution", format, backend), TOLERANCE, (format, backend))

    def test_every_entry_in_every_format(self):
        for golden in GOLDEN:
            for format, backend in READERS:
                with self.subTest(golden.name, format=format, backend=backend):
                    self.assertLess(outcome(golden.name, format, backend), TOLERANCE)


if __name__ == "__main__":
    unittest.main()

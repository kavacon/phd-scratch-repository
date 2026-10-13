"""
The shared packages must not depend on experiments, and each layer only on the ones below it. Inside `compiler`,
the core (grammar, parser, instruction set, printer) does not know which passes exist:

    compiler  <  checking  <  demo  <  experiments

An experiment may use any shared package; one experiment must not use another's code.
"""
import ast
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
LAYERS = ["compiler", "checking", "demo", "experiments"]


def imported_packages(path: Path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module
            yield from (f"{node.module}.{alias.name}" for alias in node.names)


class LayeringTestCase(unittest.TestCase):
    def test_no_layer_imports_one_above_it(self):
        for rank, layer in enumerate(LAYERS):
            above = set(LAYERS[rank + 1:])
            for path in (SRC / layer).rglob("*.py"):
                for name in imported_packages(path):
                    self.assertNotIn(name.split(".")[0], above, f"{path.relative_to(SRC)} imports {name}")

    def test_the_compiler_core_does_not_import_passes(self):
        for path in (SRC / "compiler").rglob("*.py"):
            if "passes" in path.relative_to(SRC / "compiler").parts:
                continue
            for name in imported_packages(path):
                self.assertFalse(name.startswith("compiler.passes"), f"{path.relative_to(SRC)} imports {name}")

    def test_an_experiment_does_not_import_another_experiment(self):
        for experiment in (SRC / "experiments").iterdir():
            if not experiment.is_dir() or experiment.name.startswith("__"):
                continue
            for path in experiment.rglob("*.py"):
                for name in imported_packages(path):
                    parts = name.split(".")
                    if parts[0] == "experiments" and len(parts) > 1:
                        self.assertEqual(experiment.name, parts[1], f"{path.relative_to(SRC)} imports {name}")


if __name__ == "__main__":
    unittest.main()

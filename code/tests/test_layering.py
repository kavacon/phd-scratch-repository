"""
The shared packages must not depend on experiments, and each layer only on the ones below it. Inside `compiler`,
the core (grammar, parser, instruction set, printer) does not know which passes exist:

    compiler  <  checking  <  demo  <  experiments

The first two live in `helper/python`, the demo framework in `demo/python`, and each experiment in its own
`experiment_<name>/python`. An experiment may use any shared package; one experiment must not use another's code.
"""
import ast
import unittest
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
HELPER = ["compiler", "checking"]
DEMO = ["demo"]
EXPERIMENTS = {package.name: package for root in sorted(CODE.glob("experiment_*/python")) for package in root.iterdir()
               if package.is_dir() and not package.name.startswith(("__", "."))}


def imported_packages(path: Path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module
            yield from (f"{node.module}.{alias.name}" for alias in node.names)


def package_dir(name: str) -> Path:
    return CODE / ("helper" if name in HELPER else "demo") / "python" / name


class LayeringTestCase(unittest.TestCase):
    def test_no_layer_imports_one_above_it(self):
        layers = [HELPER[:1], HELPER[1:], DEMO, list(EXPERIMENTS)]
        for rank, layer in enumerate(layers):
            above = {name for later in layers[rank + 1:] for name in later}
            for name in layer:
                directory = EXPERIMENTS[name] if name in EXPERIMENTS else package_dir(name)
                for path in directory.rglob("*.py"):
                    for imported in imported_packages(path):
                        self.assertNotIn(imported.split(".")[0], above, f"{path.relative_to(CODE)} imports {imported}")

    def test_the_compiler_core_does_not_import_passes(self):
        compiler = package_dir("compiler")
        for path in compiler.rglob("*.py"):
            if "passes" in path.relative_to(compiler).parts:
                continue
            for name in imported_packages(path):
                self.assertFalse(name.startswith("compiler.passes"), f"{path.relative_to(CODE)} imports {name}")

    def test_an_experiment_does_not_import_another_experiment(self):
        for name, directory in EXPERIMENTS.items():
            for path in directory.rglob("*.py"):
                for imported in imported_packages(path):
                    other = imported.split(".")[0]
                    self.assertTrue(other == name or other not in EXPERIMENTS, f"{path.relative_to(CODE)} imports {imported}")


if __name__ == "__main__":
    unittest.main()

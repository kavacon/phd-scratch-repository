"""
A source program taken through the pipeline: translated to `func` and `arith`, optimised by xDSL's standard passes,
lowered to QC by our pattern pass, handed to MQT, exported as OpenQASM 3 and run on Qiskit Aer, with the source
interpreter's value for comparison. The passes are one list, so another experiment is another list.

    PYTHONPATH=src .venv/bin/python -m experiment_existing_ir.experiment
"""
from xdsl.passes import PassPipeline
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.dead_code_elimination import DeadCodeElimination

from checking.simulate import simulate
from compiler.parser.parser import parse as parse_source
from experiment_existing_ir.ir import context, show
from experiment_existing_ir.ranges import ranges
from experiment_existing_ir.rewrites.to_qc import LowerToQC
from experiment_existing_ir.simulate import execute, mqt
from experiment_existing_ir.translate import translate

SOURCE = "x = 5; y = x + 3; y = y - 1; return y;"

PASSES = (CanonicalizePass(), DeadCodeElimination(), LowerToQC())


def show_between(done, module, upcoming) -> None:
    """Called before the first pass and after each one: the program and the range of each value."""
    if isinstance(done, LowerToQC):  # printed in full below, as MQT reads it
        return
    print(f"\nafter {done.name if done else 'translation'}:\n{show(module)}")
    found = ranges(module)
    for op in module.walk():
        if op.results and op.results[0] in found:
            print(f"  {op.name:15} range {tuple(found[op.results[0]])} -> {found[op.results[0]].width} qubits")


def main() -> None:
    print(f"source: {SOURCE}\nsource interpreter: {simulate(parse_source(SOURCE)).value}")
    module = translate(parse_source(SOURCE))
    PassPipeline(PASSES, show_between).apply(context(), module)
    print(f"\nlowered to QC, as MQT prints it:\n{mqt(module).ir}")
    outcome = execute(module)
    print(f"\nrun on Qiskit Aer via OpenQASM 3: value {outcome['value']}, qubits left set outside the result "
          f"{outcome['leaked']}")


if __name__ == "__main__":
    main()

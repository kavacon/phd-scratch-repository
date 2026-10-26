"""
Source programs taken through the pipeline: translated to MLIR text, optimised and lowered to QC by the C++ tool,
handed to MQT, exported as OpenQASM 3 and run on Qiskit Aer. The passes are one list, so another experiment is another
list. The tool prints the program after every pass, which shows what each one did.

    . ./env.sh && .venv/bin/python -m experiment_existing_ir.experiment

Needs the tool built first: `helper/cpp/setup.sh`, then `experiment_existing_ir/cpp/build.sh`.
"""
import subprocess

from compiler.parser.parser import parse
from experiment_existing_ir import simulate
from experiment_existing_ir.translate import translate

PASSES = ("--canonicalize", "--lower-to-qc")
PROGRAMS = {
    "arithmetic": ("x = 5; y = x + 3; y = y - 1; return y;", "v1"),
    "conditional on a distribution": ("a = 0..4; x = 2; if a < 2 { x += 3; } else { x -= 1; } return x;", "v2"),
}


def main() -> None:
    for title, (source, grammar) in PROGRAMS.items():
        program = translate(parse(source, grammar))
        print(f"\n==== {title}\n{source}\n\ntranslated:\n{program}")
        stages = subprocess.run([simulate.TOOL, *PASSES, "--mlir-print-ir-after-all"], input=program, capture_output=True,
                                text=True, check=True).stderr
        print(f"\nafter each pass:\n{stages}")
        print(f"run on Qiskit Aer via OpenQASM 3: {simulate.run(program, PASSES)}")


if __name__ == "__main__":
    main()

# Working in `python/`

A small quantum compiler and the experiments built on it. Read this before changing anything. Everything is run
from this directory with `PYTHONPATH=src`.

```
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t .     # all tests, ~10 s
PYTHONPATH=src .venv/bin/python -m experiments.allocation_poc.experiment --program area [--stages]
PYTHONPATH=src .venv/bin/python -m experiments.allocation_poc.demos.explorer        # web app on :8000
PYTHONPATH=src .venv/bin/python -m experiments.allocation_poc.demos.compare         # needs `pip install '.[circuits,compare]'`
```

## Layout and the one rule

```
src/compiler/        the compiler: parser, instruction set, printer, System, Compiler, and passes/
src/checking/        reference machines and checks that are not part of the compiler (simulate, circuits, qiskit_tools)
src/demo/            the demo-app framework: model.py (what a demo describes), app.py (server), static/ (the one page)
src/experiments/<name>/   one experiment: experiment.py, its passes/, its demos/, anything else specific to it
tests/               mirrors src/
```

Layers: `compiler < checking < demo < experiments`. A package imports only from layers to its left; one experiment never
imports another; `compiler/` outside `compiler/passes/` never imports a pass. `tests/test_layering.py` enforces it.
**If a second experiment needs something, move it down a layer; do not import across experiments.**

## The compiler

A program moves through three levels of representation, one direction only:

| level | what it is | made by |
|---|---|---|
| `NODE` | the typed AST (a list of nodes), `compiler/parser/grammar.py` | `parse(source)` |
| `ISA` | a flat `IsaProgram` of instructions (`ALLOC`, `MUL`, `CTRL`, `SCRATCH`, ...), `compiler/parser/instructions.py` | `lower_to_isa` |
| `CIRCUIT` | a Qiskit `QuantumCircuit`, one register per slot plus `scratch` | `to_openqasm` |

A pass is a function `(program, system=None) -> program` tagged with the levels it takes and gives. It stays on its
level or moves to the next, so a pipeline is: node passes, one lowering to ISA, ISA passes, one lowering to a circuit,
circuit passes. `System(register_width, qubits)` says what machine to compile for and goes to every pass.

```python
from compiler.compiler import ISA, NODE, Compiler, compile_pass
from compiler.system import System

@compile_pass(NODE, NODE, title="Drop dead code", summary="Removes assignments nothing reads.", requires=("rename",))
def drop_dead_code(program, system=None):
    ...
    return program

compiler = Compiler(System(register_width=8, qubits=200)).load(inline, unroll, drop_dead_code, lower_to_isa, allocate, to_openqasm)
result = compiler.run(source)             # result.stages: (name, level, output, warnings) for every pass, source first
result["allocate"]                        # one pass's output          result.circuit   # the last output, if a circuit
compiler.run(source, until="allocate")    # stop early                 compiler.run(source, skip=["unroll"])
```

- Passes are plain functions: call them directly in tests (`inline(nodes)`). `@compile_pass` only tags them.
- `title`, `summary` and `requires` on the decorator are how a viewer names and explains a pass (and warns when one
  it cannot work without has been switched off). Declare `requires` only for a real dependency (the pass refuses
  the program, or gives a wrong one, without it), never for a preferred order, and never for the level a pass takes:
  the compiler already enforces levels. Demos read them from the compiler (`fn.compile_pass`, `stage.title`); do not
  keep a separate table of step names in a demo.
- `load` checks the chain connects and names are unique. `run` raises `PassError(name, stages, cause, partial)`
  when a pass fails: `stages` is everything before it and `partial` is whatever it could still show.
- Pass names are function names; keep them unique per compiler.
- **Registers.** `allocate` returns an ordinary `IsaProgram` on physical registers `R0, R1, ...` with `SCRATCH R<n>`
  first, naming the scratch register (always after the slots). The code generator gets the layout from that and from
  `system.register_width`. Do not reintroduce a side object carrying slot counts; they are derivable.
- **Circuits.** The circuit is a Qiskit `QuantumCircuit`; there is no private gate list. `openqasm_text(circuit)` prints
  it (X gates with `ctrl`/`negctrl`, falling back to Qiskit's printer for anything else). `Gate` in `qasm.py` is only the
  builder's notation while one instruction is being generated. The optimiser (`compiler/passes/optimise.py`) is a
  circuit-to-circuit pass like any other.
- Reference machines (`checking.simulate`): `simulate` runs nodes, `simulate_isa` runs instructions, and
  `execute(output)` picks by type (None for circuits). Use them to check a pass keeps the result.
- Qiskit is a hard dependency. Aer (simulator runs) and `qiskit-qasm3-import` (reading text back) are extras.

## Adding an experiment

1. `src/experiments/<name>/__init__.py` (docstring: what question it asks) and `tests/experiments/<name>/__init__.py`.
2. Put what is specific to it in `src/experiments/<name>/passes/` as `@compile_pass` functions. Shared passes are in
   `compiler/passes/`; use them by importing.
3. `experiment.py`: plain imports, the pass order, a compiler, a `main()`. No web code. Pattern:

   ```python
   PIPELINE = (inline, unroll, my_pass, lower_to_isa, allocate, to_openqasm, optimise)

   def compiler(system=None): return Compiler(system).load(*PIPELINE)
   def compile_circuit(source, system=None): return compiler(system).run(source, until="to_openqasm").circuit
   ```
4. Tests check the compiler through `compiler(...).run(...)` and `checking.simulate.execute`, never through web output.
5. If the output must not drift while refactoring, pin it like `tests/experiments/allocation_poc/test_pinned.py`
   (hash of the generated text plus gate/depth/wire counts). Update a pin only on purpose, and say why.

## Adding a demo (web app)

A demo is a `demo.app.Demo` in `experiments/<name>/demos/<app>.py`. It describes; the page draws.

```python
DEMO = app.Demo(
    title="...", description="...",
    run=run,                               # (inputs: dict, enabled_steps: list[str]) -> Result
    inputs=(Text("source", "Source"), Number("register_width", "Qubits per register", 6, 1, 64)),
    cases=(Case("area", "A temporary", "what it shows", {"source": "..."}),),
    steps=tuple(Step(s.name, s.title, s.summary, s.requires) for s in specs),   # optional: switchable stages, from the passes
    actions={"simulate": simulate},        # (inputs, steps, payload) -> blocks, reached by an `Action` block
    workflow=True,                         # False: show all stages one under another
    theme=Theme(accent="#0f766e", accent_soft="#ccfbf1"),   # optional colours; the shared look when omitted
)
if __name__ == "__main__":
    app.serve(DEMO)
```

`run` returns `Result(stages=(Stage(id, name, summary, blocks),))`, or `Result(error=SourceError(...))` for bad input.
Blocks are in `demo/model.py`: `Code`, `Diff` (`Diff.of(old, new)` marks the changes), `Metrics`, `Stats`, `Switcher`,
`Grid`, `Timeline`, `Note`, `Action`, `Diagram`, `Columns`. `key=` on lines, cards and grid columns links elements:
hovering one highlights the others with the same key.

- **Colours** are a `demo.model.Theme` on the `Demo`: accent, diff colours, good/bad, the two series, and the
  surface colours (background, panel, text, muted, border). Anything left out keeps the shared look. Brand colours
  apply in both light and dark; surface colours apply in light only unless `dark=Theme(...)` gives them, because a
  light background needs dark text. The page sets them as CSS variables, so every block (and the circuit
  drawing) follows.
- **Never write HTML or JavaScript in a demo.** If a display is missing, add a block type in `demo/model.py` and a
  renderer in `demo/static/app.js`, so every demo gets it.
- Keep compile logic out of the demo: it calls `compiler(...).run(...)`, then describes what came back. The explorer
  builds its stages from `result.stages`; mirror that.
- Servers handle one request at a time on the main thread on purpose (Qiskit's native allocator crashes when threads
  that used it end). Do not switch to a threading server.
- Test demos only for shape: that every block it returns is a known type and serialises (`tests/experiments/*/test_demos.py`).

## Conventions

- Integers are two's complement and wrap modulo 2 to the register width everywhere (reference machines and circuits).
  Add, subtract and multiply are the same gates as unsigned; only `<`, `>` and how a value is read differ. Pass the
  system to `simulate`, `simulate_isa` and `execute` so the reference wraps the way the circuit does, and use
  `checking.circuits.to_signed` to read a measured register.
- The language has negative literals and negation (`-x` is `0 - x`), and in-place `x += e;` and `x -= e;`. An update is
  an assignment marked `inplace`: rename still gives it a new version, and `lower_to_isa` makes that version another
  name for the old register and emits an `Update` instruction (`ADD=`, `SUB=`). Only add and subtract, because
  only they can be undone in place; `x += x` is refused. Release insertion (the allocation experiment) cannot undo
  an update and refuses a program that has one.
- Division and squaring a register have no circuit and are refused by the code generator.
- Match the surrounding code's comment density and naming. Docstrings say why, not what.
- When something about a pass's behaviour or an experiment's result is a decision, record the rationale in the
  experiment's Notion page (the repo is not where decisions live).

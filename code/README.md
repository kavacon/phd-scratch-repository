# code

Compiler proof of concept and the experiments built on it.

```
helper/python/
  compiler/      shared: the Compiler and System, grammar, parser (and its syntax-error messages), instruction set, printer
    passes/      passes any experiment can use: inline, unroll, flatten, balance, rename, ISA lowering, physical allocation,
                 circuit generation (a Qiskit circuit), Qiskit optimisation (the core above does not import them)
  checking/      shared: reference machines (AST and instruction level), gate-level checks, Qiskit/Aer helpers
helper/resources/  the grammars (grammar_v1.lark, grammar_v2.lark), read by the parser and by anything else that parses the language
demo/python/demo/  shared: the demo app framework. model.py is the data a demo fills in, app.py serves it. Also HTTP server,
                 diffing, circuit drawing
demo/resources/  the one generic page that draws a demo (app.html, app.js, app.css, circuit.js)
experiment_allocation_poc/python/experiment_allocation_poc/
      experiment.py     the pass order, a Compiler, compile_source / compile_circuit, and a CLI. No web code.
      passes/           liveness, release insertion
      demos/            explorer.py (+ explorer_blocks.py) and compare.py: web apps built on `demo`
      programs.py, views.py, qrisp_programs.py, qrisp_comparison.py
experiment_encodings/python/experiment_encodings/
                        Same program, different encodings: two binary, two Fourier and a residue number system lowering. experiment.py (the
                        --encoding flag, pipelines, outcomes), encodings.py (the registry), passes/ (one lowering each,
                        and optimise_zx), programs.py, isolated.py, readout.py, demos/comparison.py, demos/instructions.py
experiment_distributions/python/experiment_distributions/
                        Working with distributions: what superposition and entanglement mean for the language, what the compiler converts to
                        superposition, and uncomputation of what it entangles. experiment.py, passes/, programs.py, demos/ (scaffold only)
experiment_existing_ir/python/experiment_existing_ir/
                        Whether an MLIR-based stack can carry the language to circuits: translation to `func` and `arith`, an xDSL
                        lowering to MQT's QC dialect, range analysis. experiment.py, translate.py, rewrites/, dialects/
<folder>/tests/  each folder's tests (helper/tests/test_layering.py checks the layering of all of them)
```

Every folder keeps its Python in `python/`, any C++ in `cpp/`, and non-code files in `resources/` (or beside the experiment,
like `experiment_encodings/results/`), so none of them depend on a language.

## Layering

`compiler` < `checking` < `demo` < `experiment_*`. A package may import from the ones to its left, never the ones
to its right, and one experiment never imports another. Inside `compiler`, the core does not import `passes`.
`helper/tests/test_layering.py` enforces this.

An experiment is: `experiment.py` that runs it with plain imports and a `main()`; modules specific to it
(`passes/`, programs, comparisons); and `demos/` with one file per web app. When a second experiment needs
something an experiment defined, move it down into `helper` or `demo`.

## Demos

A demo is a `demo.app.Demo`: inputs (`Text`, `Number`), cases, steps the viewer can switch on and off, a `run`
function and optional actions. `run(inputs, enabled_steps)` returns a `Result`: stages, each a list of blocks. The page
draws the blocks and nothing else, so demos never contain HTML or JavaScript. The blocks are `code`, `diff`,
`metrics`, `stats`, `switcher`, `grid`, `timeline`, `note`, `action`, `diagram` and `columns` (see `demo/model.py`).
A new kind of display is added once in `demo/model.py` and `demo/resources/app.js`, for every demo to use.

The page loads `/api/bootstrap` (title, inputs, cases, steps, action urls), posts the inputs to `/api/run`, and an
`action` block posts to the url the bootstrap gave it and draws the blocks that come back.

Tests check the compiler through `compiler(...).run(...)` (what every pass produced) and `checking.simulate.execute`
(how it runs), not through the web output; the demo tests only check that what a demo describes is well formed.

## Run

```
. ./env.sh      # every folder's python/ on the path
.venv/bin/python -m experiment_allocation_poc.experiment --program area [--stages]
.venv/bin/python -m experiment_allocation_poc.demos.explorer
.venv/bin/python -m experiment_allocation_poc.demos.compare   # needs the compare extra
./test.sh       # all tests
pip install '.[circuits,compare]'
```

## Adding an experiment

1. Create `experiment_<name>/python/experiment_<name>/` and `experiment_<name>/tests/` (each with `__init__.py`).
2. Add `experiment.py` with a `main()`. A pass is a function from a list of AST nodes to a list of AST nodes:
   experiment-specific ones go in `experiment_<name>/python/experiment_<name>/passes/`, shared ones are in
   `helper/python/compiler/passes/`.
3. For a web app add `demos/<app>.py` that builds a `Demo` and calls `demo.app.serve`.

See `AGENTS.md` for how to use the compiler and how to set up an experiment or a demo.

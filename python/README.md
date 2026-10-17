# python

Compiler proof of concept and the experiments built on it.

```
src/
  compiler/      shared: the Compiler and System, grammar, parser (and its syntax-error messages), instruction set, printer
    passes/      passes any experiment can use: inline, unroll, flatten, balance, rename, ISA lowering, physical allocation,
                 circuit generation (a Qiskit circuit), Qiskit optimisation (the core above does not import them)
  checking/      shared: reference machines (AST and instruction level), gate-level checks, Qiskit/Aer helpers
  demo/          shared: the demo app framework. model.py is the data a demo fills in, app.py serves it,
                 static/ is the one generic page that draws it. Also HTTP server, diffing, circuit drawing
  experiments/
    allocation_poc/
      experiment.py     the pass order, a Compiler, compile_source / compile_circuit, and a CLI. No web code.
      passes/           liveness, release insertion
      demos/            explorer.py (+ explorer_blocks.py) and compare.py: web apps built on `demo`
      programs.py, views.py, qrisp_programs.py, qrisp_comparison.py
    encodings/          Same program, different encodings: two binary and two Fourier lowerings. experiment.py (the
                        --encoding flag, pipelines, outcomes), encodings.py (the registry), passes/ (one lowering each,
                        and optimise_zx), programs.py, isolated.py, readout.py, demos/comparison.py, demos/instructions.py
tests/           mirrors src/
```

## Layering

`compiler` < `checking` < `demo` < `experiments`. A package may import from the ones to its left, never the ones
to its right, and one experiment never imports another. Inside `compiler`, the core does not import `passes`.
`tests/test_layering.py` enforces this.

An experiment is: `experiment.py` that runs it with plain imports and a `main()`; modules specific to it
(`passes/`, programs, comparisons); and `demos/` with one file per web app. When a second experiment needs
something an experiment defined, move it down into `compiler`, `checking` or `demo`.

## Demos

A demo is a `demo.app.Demo`: inputs (`Text`, `Number`), cases, steps the viewer can switch on and off, a `run`
function and optional actions. `run(inputs, enabled_steps)` returns a `Result`: stages, each a list of blocks. The page
draws the blocks and nothing else, so demos never contain HTML or JavaScript. The blocks are `code`, `diff`,
`metrics`, `stats`, `switcher`, `grid`, `timeline`, `note`, `action`, `diagram` and `columns` (see `demo/model.py`).
A new kind of display is added once in `demo/model.py` and `demo/static/app.js`, for every demo to use.

The page loads `/api/bootstrap` (title, inputs, cases, steps, action urls), posts the inputs to `/api/run`, and an
`action` block posts to the url the bootstrap gave it and draws the blocks that come back.

Tests check the compiler through `compiler(...).run(...)` (what every pass produced) and `checking.simulate.execute`
(how it runs), not through the web output; the demo tests only check that what a demo describes is well formed.

## Run

```
PYTHONPATH=src .venv/bin/python -m experiments.allocation_poc.experiment --program area [--stages]
PYTHONPATH=src .venv/bin/python -m experiments.allocation_poc.demos.explorer
PYTHONPATH=src .venv/bin/python -m experiments.allocation_poc.demos.compare   # needs the compare extra
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t .
pip install '.[circuits,compare]'
```

## Adding an experiment

1. Create `src/experiments/<name>/` and `tests/experiments/<name>/` (each with `__init__.py`).
2. Add `experiment.py` with a `main()`. A pass is a function from a list of AST nodes to a list of AST nodes:
   experiment-specific ones go in `experiments/<name>/passes/`, shared ones are in `compiler/passes/`.
3. For a web app add `demos/<app>.py` that builds a `Demo` and calls `demo.app.serve`.

See `AGENTS.md` for how to use the compiler and how to set up an experiment or a demo.

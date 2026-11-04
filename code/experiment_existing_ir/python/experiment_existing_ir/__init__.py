"""
Existing IRs as the compilation stack: can the source language be translated into an MLIR form, rewritten there with
MLIR's own passes and ours, and handed to MQT's compiler for the quantum side, without a bespoke instruction set? See
the experiment page in Notion for the aim, the strategies and the work plan.

The Python side translates source programs to MLIR text (`translate`) and runs the result through the C++ tool and MQT
(`pipeline`). The tool, `cpp/`, holds the experiment's own pieces: the `dist` dialect, `resolve-dist` (one dataflow analysis and one rewrite that
resolve references, kinds and conditionals) and the `lower-to-qc` pass. Everything else is MLIR's.
"""

"""
Existing IRs as the compilation stack: can the source language be translated into an MLIR form, rewritten there with
MLIR's own passes and ours, and handed to MQT's compiler for the quantum side, without a bespoke instruction set? See
the experiment page in Notion for the aim, the strategies and the work plan.

The Python side translates source programs to MLIR text (`translate`) and runs the result through the C++ tool and MQT
(`simulate`). The tool, `cpp/`, holds the experiment's own pieces: the `dist` dialect and the `lower-to-qc` pass.
Everything else, from folding to range analysis, is MLIR's.
"""

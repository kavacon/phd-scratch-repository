"""
Existing IRs as the compilation stack: can the source language be translated into an MLIR-style form, rewritten there
with the standard passes and our own, and handed to MQT's compiler for the quantum side, without a bespoke instruction
set? See the experiment page in Notion for the aim, the strategies and the work plan.

Source programs are translated to the standard `arith`, `func` and `scf` dialects. Custom operations are added only for
constructs whose meaning differs from the classical one (creating a distribution, release regions). The passes are
written in xDSL, a Python equivalent of MLIR (not MLIR itself). Shared code (parser, source interpreter) lives in
`compiler` and `checking`; only what is specific to this experiment belongs here.
"""

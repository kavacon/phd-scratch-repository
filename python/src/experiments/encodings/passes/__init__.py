"""
Each encoding's lowering from instructions to a circuit, one module per encoding. A lowering is an ordinary
`ISA -> CIRCUIT` pass (see `compiler.compiler`): it is independent of the others, and the experiment picks one per
compilation. Binary with the Cuccaro adder is the shared `compiler.passes.qasm.to_openqasm` and has no module here.

Binary with the ancilla-free adder and Fourier are built. fourier_dual and angle are placeholders: they name the
lowering, say what it will do, and raise `NotImplementedError`. optimise_zx is not a lowering but a second way to optimise
the circuit one builds.
"""

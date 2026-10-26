"""
General outline of the POC
- Write a program that has non-trivial allocation and deallocation ordering
- Compare the verbosity and complexity scores of it compared to a non-implicit example (check article notes for complexity scoring approach)
- Using one or more (depending on the purpose of the example) known allocation algorithms compile to a circuit or circuit-ISA
- Demonstrate the circuit runs correctly as a simulation and uses efficient allocation ordering

The purpose of the demonstration is to show that it is feasible to include the feature required of implicitness and potentially
demonstrate pluggability to prove the benefits of higher level abstraction in the language. It should show:
- A fragment of the language design philosophy can be supported and is practical
- The design philosophy is compatible with state-of-the-art approaches to quantum language design
- Language is compilable and runnable

This will not:
- Be the final syntax of the language
- Guarantee type safety or correctness
- Be the final compiler

The demonstration is a POC and upcoming work would seek to use a known ISA or IR to improve portability.
"""
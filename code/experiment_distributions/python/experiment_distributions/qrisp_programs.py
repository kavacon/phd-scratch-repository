"""
The programs written by hand in Qrisp, for comparison with our compiler (see qrisp_comparison.py).

Each is the Qrisp a person would naturally write for the same task, using only what Qrisp provides: `QuantumFloat` registers,
its own `+`, `-`, `*` and comparison operators, `h` or `init_state` to put a register in superposition, `with condition:` for an
`if`, and `@auto_uncompute` for the temporaries, where our compiler inserts the releases. Constants are Python numbers, as they
would be written. The registers that hold a distribution are made in `main`, which is not uncomputed, because the result is
entangled with them and cannot be uncomputed; the work is in a function that is.

Registers are `QuantumFloat(WIDTH)`, the same width our circuits use. Qrisp sizes the results of arithmetic itself, so an
out-of-place product has more qubits than a register, and its in-place `+=` and `-=` stay in the register. A `QuantumFloat` is
a mantissa times a power of two and the comparison reads the exponent too.
"""
from typing import Dict

from checking.qrisp_tools import QrispCircuit, build

SOURCES: Dict[str, str] = {
    "arithmetic": '''from qrisp import QuantumFloat, auto_uncompute, h

WIDTH = 0

@auto_uncompute
def f(a):
    b = a + 3
    return b + b


def main():
    a = QuantumFloat(WIDTH)
    h(a[:2])
    return f(a)
''',
    "mixed": '''from qrisp import QuantumFloat, auto_uncompute

WIDTH = 0

@auto_uncompute
def f(a):
    b = a + 1
    k = 1
    m = k * 2
    c = m + 1
    return b + c


def main():
    a = QuantumFloat(WIDTH)
    a.init_state({0: 1, 1: 1, 2: 1})
    return f(a)
''',
    "temporary": '''from qrisp import QuantumFloat, auto_uncompute, h

WIDTH = 0

@auto_uncompute
def f(a):
    t = a * 3
    return t < 5


def main():
    a = QuantumFloat(WIDTH)
    h(a[:2])
    return f(a)
''',
    "fused": '''from qrisp import QuantumFloat, auto_uncompute, h

WIDTH = 0

@auto_uncompute
def f(a):
    t = a * 3
    return t * 5


def main():
    a = QuantumFloat(WIDTH)
    h(a[:2])
    return f(a)
''',
    "shared": '''from qrisp import QuantumFloat, auto_uncompute

WIDTH = 0

@auto_uncompute
def f(a, p):
    k = 2
    x = a + k
    y = p + k
    return x


def main():
    a = QuantumFloat(WIDTH)
    a.init_state({0: 1, 1: 1, 2: 1})
    p = QuantumFloat(WIDTH)
    p.init_state({0: 1, 1: 1})
    return f(a, p)
''',
    "branch-copy": '''from qrisp import QuantumFloat, auto_uncompute, h

WIDTH = 0

@auto_uncompute
def f(a):
    b = QuantumFloat(WIDTH)
    with a == 1:
        b += 1
    return b


def main():
    a = QuantumFloat(WIDTH)
    h(a[0])
    return f(a)
''',
    "branch-arms": '''from qrisp import QuantumFloat, auto_uncompute, h

WIDTH = 0

@auto_uncompute
def f(a):
    x = QuantumFloat(WIDTH)
    x[:] = 2
    small = a < 2
    with small:
        x += 3
    with ~small:
        x -= 1
    return x


def main():
    a = QuantumFloat(WIDTH)
    h(a[:2])
    return f(a)
''',
    "branch-nested": '''from qrisp import QuantumFloat, auto_uncompute, h

WIDTH = 0

@auto_uncompute
def f(a, b):
    x = QuantumFloat(WIDTH)
    with a == 1:
        with b == 1:
            x += 3
        with b == 0:
            x += 1
    return x


def main():
    a = QuantumFloat(WIDTH)
    h(a[0])
    b = QuantumFloat(WIDTH)
    h(b[0])
    return f(a, b)
''',
    "branch-temporary": '''from qrisp import QuantumFloat, auto_uncompute

WIDTH = 0

@auto_uncompute
def f(a):
    x = QuantumFloat(WIDTH)
    x[:] = 1
    with a > 0:
        t = a * 3
        x += t
    return x


def main():
    a = QuantumFloat(WIDTH)
    a.init_state({0: 1, 1: 1, 2: 1})
    return f(a)
''',
}


def source_text(program_id: str, width: int) -> str:
    """The program as it runs, with the register width filled in."""
    return SOURCES[program_id].replace("WIDTH = 0", f"WIDTH = {width}", 1)


def run(program_id: str, width: int) -> QrispCircuit:
    """Run a Qrisp program and compile it to a Qiskit circuit with the returned register measured."""
    return build(source_text(program_id, width))

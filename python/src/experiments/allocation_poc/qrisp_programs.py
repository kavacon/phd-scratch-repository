"""
The demo programs written by hand in Qrisp, for comparison with our compiler (see compare.py).

Each is the Qrisp a person would naturally write for the same task, using only what Qrisp provides: `QuantumFloat`
registers, its own `+`, `-`, `*` and comparison operators, `with condition:` for an `if`, and `@auto_uncompute` for the
temporaries, where our compiler inserts the releases. Nothing is wrapped or annotated, so the circuits are what
Qrisp's compiler makes of ordinary Qrisp code, and the source text shown in the app is exactly what runs.

Registers are `QuantumFloat(WIDTH)`, the same width our circuits use. Qrisp sizes the results of arithmetic itself (they
grow rather than wrapping modulo 2 to the width), picks its own adders, and allocates and recycles its own qubits. A
register loaded with a constant is not cleaned up by `@auto_uncompute`, so it is still set when the circuit ends (the app
shows those as qubits left set when a circuit is run). A `QuantumFloat` is a mantissa times a power of two, and
multiplying by 2 only shifts the exponent, after which Qrisp refuses to add an odd constant (its precision is too
low), so doubling is written `a + a`. Where our source assigns a name again inside an `if`, the Qrisp updates one
register in place under the condition, but copies first where that would stop Qrisp uncomputing the condition.
"""
import contextlib
import io
from dataclasses import dataclass
from typing import Dict, Tuple

SOURCES: Dict[str, str] = {
    "area": '''from qrisp import QuantumFloat, auto_uncompute

WIDTH = 0

@auto_uncompute
def area(w, h):
    p = w * h
    return p + 1


@auto_uncompute
def main():
    x = QuantumFloat(WIDTH)
    x[:] = 3
    y = QuantumFloat(WIDTH)
    y[:] = 4
    return area(x, y)
''',
    "chain": '''from qrisp import QuantumFloat, auto_uncompute

WIDTH = 0

@auto_uncompute
def f(x):
    a = x + 1
    b = a + a
    return b + a


@auto_uncompute
def main():
    x = QuantumFloat(WIDTH)
    x[:] = 3
    return f(x)
''',
    "expression": '''from qrisp import QuantumFloat, auto_uncompute

WIDTH = 0

@auto_uncompute
def f(a):
    t = a + 1
    u = t * 3
    return t + u


@auto_uncompute
def main():
    a = QuantumFloat(WIDTH)
    a[:] = 2
    return f(a)
''',
    "if-else": '''from qrisp import QuantumFloat, auto_uncompute

WIDTH = 0

@auto_uncompute
def f(x):
    a = x + 1
    y = QuantumFloat(WIDTH)
    small = a < 5
    with small:
        t = a + a
        y += t + 1
    with ~small:
        y += a - 1
    return y + 1


@auto_uncompute
def main():
    x = QuantumFloat(WIDTH)
    x[:] = 2
    return f(x)
''',
    "one-armed": '''from qrisp import QuantumFloat, auto_uncompute

WIDTH = 0

@auto_uncompute
def f(a):
    x = a + 1
    small = x < 4
    r = QuantumFloat(WIDTH)
    r += x
    with small:
        r += 5
    return r


@auto_uncompute
def main():
    a = QuantumFloat(WIDTH)
    a[:] = 1
    return f(a)
''',
    "nested": '''from qrisp import QuantumFloat, auto_uncompute

WIDTH = 0

@auto_uncompute
def f(a):
    x = QuantumFloat(WIDTH)
    outer = a < 5
    inner = a < 3
    with outer:
        with inner:
            x += (a + 1) * 2
        with ~inner:
            x += (a + 2) * 2
    with ~outer:
        x += a - 1
    return x + a


@auto_uncompute
def main():
    a = QuantumFloat(WIDTH)
    a[:] = 1
    return f(a)
''',
    "loop": '''from qrisp import QuantumFloat, auto_uncompute

WIDTH = 0

@auto_uncompute
def g(x):
    s = x
    for i in range(3):
        t = s + i
        s = t + t + 1
    return s


@auto_uncompute
def main():
    x = QuantumFloat(WIDTH)
    x[:] = 1
    return g(x)
''',
    "width": '''from qrisp import QuantumFloat, auto_uncompute

WIDTH = 0

@auto_uncompute
def f(x):
    a = x + 1
    b = a + a
    y = b + 3
    c = x + 5
    d = c * 3
    e = d + 7
    return y


@auto_uncompute
def main():
    x = QuantumFloat(WIDTH)
    x[:] = 2
    return f(x)
''',
    "explicit": '''from qrisp import QuantumFloat, auto_uncompute

WIDTH = 0

@auto_uncompute
def f(a):
    t = a + 1
    u = t * 2
    v = u + t
    t.uncompute()
    return v


@auto_uncompute
def main():
    a = QuantumFloat(WIDTH)
    a[:] = 2
    return f(a)
''',
    "calls": '''from qrisp import QuantumFloat, auto_uncompute

WIDTH = 0

@auto_uncompute
def double(x):
    y = x * 2
    return y + 0


@auto_uncompute
def quad(x):
    d = double(x)
    return double(d)


@auto_uncompute
def main():
    x = QuantumFloat(WIDTH)
    x[:] = 3
    return quad(x)
''',
}


@dataclass(frozen=True)
class QrispCircuit:
    circuit: object  # a Qiskit QuantumCircuit, with the returned register measured
    result: Tuple[int, ...]  # the qubit of each bit of the returned register (its mantissa), least significant first
    exponent: int  # a QuantumFloat holds mantissa * 2 ** exponent
    labels: Tuple[str, ...]  # a label per wire, from the Qrisp variable it belongs to
    groups: Tuple[int, ...]  # which variable each wire belongs to, to colour labels


def source_text(program_id: str, width: int) -> str:
    """The program as it runs, with the register width filled in."""
    return SOURCES[program_id].replace("WIDTH = 0", f"WIDTH = {width}", 1)


def run(program_id: str, width: int) -> QrispCircuit:
    """Run a Qrisp program and compile it to a Qiskit circuit with the returned register measured."""
    from qiskit import ClassicalRegister

    namespace: Dict[str, object] = {}
    exec(source_text(program_id, width), namespace)  # trusted: the programs above, exactly what the app displays
    with contextlib.redirect_stdout(io.StringIO()):  # Qrisp prints progress while it simulates
        result = namespace["main"]()
        compiled = result.qs.compile(intended_measurements=list(result.reg))
        indices = tuple(compiled.qubits.index(q) for q in result.reg)
        circuit = compiled.to_qiskit()
        exponent = int(result.exponent)
    labels, groups, seen = [], [], {}
    for qubit in circuit.qubits:
        registers = circuit.find_bit(qubit).registers
        label = registers[0][0].name if registers else str(len(labels))
        labels.append(label)
        groups.append(seen.setdefault(label.split(".")[0], len(seen)))
    circuit.add_register(ClassicalRegister(len(indices), "result"))
    for position, qubit in enumerate(indices):
        circuit.measure(qubit, position)
    return QrispCircuit(circuit, indices, exponent, tuple(labels), tuple(groups))

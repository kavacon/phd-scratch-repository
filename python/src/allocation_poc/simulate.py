"""
Register machine for programs after rename (it also runs unrenamed programs). It checks what the insertion
pass promises:
- every `x ~= w;` finds x holding the value of w, and then frees x,
- nothing reads a register that was never defined or has been freed,
- the registers left over when a function or the program returns (`leaks`) are only what is expected.

Each version is a register. `Assign` writes one, `~=` frees one, loop headers copy the entry or back-edge
version into the header register. Registers a function returns or takes as parameters are not leaks.
It also records the most registers held at once in each scope (`peak`), a stand-in for circuit width.
"""
import operator
from typing import Any, Dict, List, NamedTuple

from parser.grammar import (
    Assign, BinOp, Boolean, Call, Equals, ExprStmt, ForLoop, FunctionDef, If, Node, Not, Number, Return,
    Unassign, Variable,
)


class SimulationError(AssertionError):
    pass


_OPS = {
    "+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv,
    "<": operator.lt, ">": operator.gt, "!=": operator.ne,
    "&&": lambda a, b: a and b, "||": lambda a, b: a or b,
}

MAX_STEPS = 200_000


class _Return:
    def __init__(self, value: Any, register: str = ""):
        self.value, self.register = value, register


class Result(NamedTuple):
    value: Any
    leaks: Dict[str, set]  # registers still held on return, by scope ("<program>" or a function name)
    peak: Dict[str, int]  # most registers held at once, by scope


class Machine:
    def __init__(self, program: List[Node]):
        self.functions = {s.name: s for s in program if isinstance(s, FunctionDef)}
        self.leaks: Dict[str, set] = {}
        self.peak: Dict[str, int] = {}
        self.steps = 0

    def run(self, program: List[Node]) -> Any:
        registers: Dict[str, Any] = {}
        outcome = self.block(program, registers, "<program>")
        self.record("<program>", registers, outcome, ())
        return outcome.value

    def record(self, scope: str, registers: Dict[str, Any], outcome: _Return, params) -> None:
        self.leaks.setdefault(scope, set()).update(set(registers) - set(params) - {outcome.register})

    def block(self, body, registers: Dict[str, Any], scope: str):
        for statement in body:
            outcome = self.statement(statement, registers, scope)
            if isinstance(outcome, _Return):
                return outcome

    def hold(self, name: str, value: Any, registers: Dict[str, Any], scope: str) -> None:
        registers[name] = value
        self.peak[scope] = max(self.peak.get(scope, 0), len(registers))

    def statement(self, node: Node, registers: Dict[str, Any], scope: str):
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise SimulationError("step limit reached (does a loop or a recursive call never end?)")
        if isinstance(node, Assign):
            self.hold(node.name, self.eval(node.value, registers, scope), registers, scope)
        elif isinstance(node, Unassign):
            if node.name not in registers:
                raise SimulationError(f"{node.name} released but not held")
            expected = self.eval(node.witness, registers, scope)
            if registers[node.name] != expected:
                raise SimulationError(f"{node.name} holds {registers[node.name]} but its witness is {expected}")
            del registers[node.name]
        elif isinstance(node, ExprStmt):
            self.eval(node.expr, registers, scope)
        elif isinstance(node, Return):
            register = node.value.name if isinstance(node.value, Variable) else ""
            return _Return(self.eval(node.value, registers, scope), register)
        elif isinstance(node, If):
            branch = node.body if self.eval(node.condition, registers, scope) else node.orelse
            return self.block(branch, registers, scope)
        elif isinstance(node, ForLoop):
            start, stop = self.eval(node.start, registers, scope), self.eval(node.stop, registers, scope)
            i, first = start, True
            while True:
                for phi in node.header:
                    entry, back = phi.value.operands
                    self.hold(phi.name, self.read((entry if first else back).name, registers), registers, scope)
                first = False
                if i >= stop:
                    break
                self.hold(node.var, i, registers, scope)
                self.block(node.body, registers, scope)
                registers.pop(node.var, None)
                i += 1
        return None

    def read(self, name: str, registers: Dict[str, Any]) -> Any:
        if name not in registers:
            raise SimulationError(f"{name} read but not held")
        return registers[name]

    def eval(self, node: Node, registers: Dict[str, Any], scope: str) -> Any:
        if isinstance(node, (Number, Boolean)):
            return node.value
        if isinstance(node, Variable):
            return self.read(node.name, registers)
        if isinstance(node, BinOp):
            return _OPS[node.op](self.eval(node.left, registers, scope), self.eval(node.right, registers, scope))
        if isinstance(node, Not):
            return not self.eval(node.operand, registers, scope)
        if isinstance(node, Equals):
            values = [self.eval(o, registers, scope) for o in node.operands]
            return all(a == b for a, b in zip(values, values[1:]))
        if isinstance(node, Call):
            if node.name not in self.functions:
                raise SimulationError(f"function {node.name} is not defined")
            function = self.functions[node.name]
            if len(node.args) != len(function.params):
                raise SimulationError(f"{node.name} expects {len(function.params)} argument(s)")
            local: Dict[str, Any] = {}
            for p, a in zip(function.params, node.args):
                self.hold(p.name, self.eval(a, registers, scope), local, node.name)
            outcome = self.block(function.body, local, node.name)
            self.record(node.name, local, outcome, [p.name for p in function.params])
            return outcome.value
        raise SimulationError(f"cannot evaluate {type(node).__name__}")


def simulate(program: List[Node]) -> Result:
    """Run a program; returns its value, the leaked registers of each scope and the peak registers of each scope."""
    machine = Machine(program)
    value = machine.run(program)
    return Result(value, machine.leaks, machine.peak)

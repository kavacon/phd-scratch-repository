"""
Register machine for programs after rename (it also runs unrenamed programs). It checks what the insertion
pass promises:
- every `x ~= w;` finds x holding the value of w, and then frees x,
- nothing reads a register that was never defined or has been freed,
- the registers left over when a function or the program returns (`leaks`) are only what is expected.

Each version is a register. `Assign` writes one and `~=` frees one. Registers a function returns or takes as
parameters are not leaks. Loops run directly, so a program can be run before it is unrolled.
It also records the most registers held at once in each scope (`peak`), a stand-in for circuit width.
"""
import operator
from typing import Any, Dict, List, NamedTuple, Optional

from compiler.parser.grammar import (
    Assign,
    BinOp,
    Boolean,
    Call,
    Equals,
    ExprStmt,
    ForLoop,
    FunctionDef,
    If,
    Node,
    Not,
    Number,
    Return,
    Unassign,
    Variable,
)
from compiler.parser.instructions import Alloc, Const, Ctrl, EndCtrl, Free, ICtrl, IOp, IsaProgram, Op, Reg, Update
from compiler.system import System


class SimulationError(AssertionError):
    pass


def _and(a, b):
    return a and b


def _or(a, b):
    return a or b


_BINARY = {
    "add": operator.add, "sub": operator.sub, "mul": operator.mul, "div": operator.truediv,
    "lt": operator.lt, "gt": operator.gt, "eq": operator.eq, "ne": operator.ne, "and": _and, "or": _or,
}
_OPS = {  # the source operators, by the opcode they lower to
    "+": _BINARY["add"], "-": _BINARY["sub"], "*": _BINARY["mul"], "/": _BINARY["div"],
    "<": _BINARY["lt"], ">": _BINARY["gt"], "!=": _BINARY["ne"], "&&": _and, "||": _or,
}

MAX_STEPS = 200_000

_ARITHMETIC = ("+", "-", "*", "add", "sub", "mul")


def normaliser(system: Optional[System]):
    """
    How integers are held on the machine `system` describes: wrapped to the register width, as two's complement
    (-2^(w-1) to 2^(w-1)-1). With no system integers are unbounded, which is the reference for programs that never
    overflow.
    """
    if system is None:
        return lambda value: value
    width = system.register_width
    mask, half = (1 << width) - 1, 1 << (width - 1)

    def normalise(value):
        if isinstance(value, bool) or not isinstance(value, int):
            return value
        value &= mask
        return value - (mask + 1) if value >= half else value
    return normalise


class _Return(NamedTuple):
    value: Any
    register: str = ""  # the register the value was read from, when it was a plain variable


class Result(NamedTuple):
    value: Any
    leaks: Dict[str, set]  # registers still held on return, by scope ("<program>" or a function name)
    peak: Dict[str, int]  # most registers held at once, by scope


class Machine:
    def __init__(self, program: List[Node], system: Optional[System] = None):
        self.norm = normaliser(system)
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
        return None

    def hold(self, name: str, value: Any, registers: Dict[str, Any], scope: str) -> None:
        registers[name] = value
        self.peak[scope] = max(self.peak.get(scope, 0), len(registers))

    def tick(self, message: str) -> None:
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise SimulationError(message)

    def statement(self, node: Node, registers: Dict[str, Any], scope: str):
        self.tick("step limit reached (does a loop or a recursive call never end?)")
        if isinstance(node, Assign):
            value = self.eval(node.value, registers, scope)
            old = node.value.left if node.inplace and isinstance(node.value, BinOp) else None
            if isinstance(old, Variable) and old.name != node.name:
                registers.pop(old.name, None)  # an update takes over the register of the version it replaces
            self.hold(node.name, value, registers, scope)
        elif isinstance(node, Unassign):
            self.release(node, registers, scope)
        elif isinstance(node, ExprStmt):
            self.eval(node.expr, registers, scope)
        elif isinstance(node, Return):
            register = node.value.name if isinstance(node.value, Variable) else ""
            return _Return(self.eval(node.value, registers, scope), register)
        elif isinstance(node, If):
            branch = node.body if self.eval(node.condition, registers, scope) else node.orelse
            return self.block(branch, registers, scope)
        elif isinstance(node, ForLoop):
            self.loop(node, registers, scope)
        return None

    def release(self, node: Unassign, registers: Dict[str, Any], scope: str) -> None:
        if node.name not in registers:
            raise SimulationError(f"{node.name} released but not held")
        expected = self.eval(node.witness, registers, scope)
        if registers[node.name] != expected:
            raise SimulationError(f"{node.name} holds {registers[node.name]} but its witness is {expected}")
        del registers[node.name]

    def loop(self, node: ForLoop, registers: Dict[str, Any], scope: str) -> None:
        start, stop = self.eval(node.start, registers, scope), self.eval(node.stop, registers, scope)
        for i in range(start, stop):
            self.hold(node.var, i, registers, scope)
            self.block(node.body, registers, scope)
            registers.pop(node.var, None)

    def read(self, name: str, registers: Dict[str, Any]) -> Any:
        if name not in registers:
            raise SimulationError(f"{name} read but not held")
        return registers[name]

    def eval(self, node: Node, registers: Dict[str, Any], scope: str) -> Any:
        if isinstance(node, Number):
            return self.norm(node.value)
        if isinstance(node, Boolean):
            return node.value
        if isinstance(node, Variable):
            return self.read(node.name, registers)
        if isinstance(node, BinOp):
            value = _OPS[node.op](self.eval(node.left, registers, scope), self.eval(node.right, registers, scope))
            return self.norm(value) if node.op in _ARITHMETIC else value
        if isinstance(node, Not):
            return not self.eval(node.operand, registers, scope)
        if isinstance(node, Equals):
            values = [self.eval(o, registers, scope) for o in node.operands]
            return all(a == b for a, b in zip(values, values[1:], strict=False))
        if isinstance(node, Call):
            if node.name not in self.functions:
                raise SimulationError(f"function {node.name} is not defined")
            function = self.functions[node.name]
            if len(node.args) != len(function.params):
                raise SimulationError(f"{node.name} expects {len(function.params)} argument(s)")
            local: Dict[str, Any] = {}
            for p, a in zip(function.params, node.args, strict=True):
                self.hold(p.name, self.eval(a, registers, scope), local, node.name)
            outcome = self.block(function.body, local, node.name)
            self.record(node.name, local, outcome, [p.name for p in function.params])
            return outcome.value
        raise SimulationError(f"cannot evaluate {type(node).__name__}")


def simulate(program: List[Node], system: Optional[System] = None) -> Result:
    """
    Run a program; returns its value, the leaked registers of each scope and the peak registers of each scope. With a
    system, integers wrap to its register width the way the circuits do.
    """
    machine = Machine(program, system)
    value = machine.run(program)
    return Result(value, machine.leaks, machine.peak)


_OPCODES = {**_BINARY, "copy": lambda a: a, "not": operator.not_}


class _IsaMachine:
    """Runs a program of instructions. `FREE` must find the register at zero, which checks every release."""

    def __init__(self, system: Optional[System] = None) -> None:
        self.norm = normaliser(system)
        self.registers: Dict[str, Any] = {}
        self.peak = 0
        self.steps = 0

    def read(self, operand):
        if isinstance(operand, Const):
            return self.norm(operand.value)
        if operand.name not in self.registers:
            raise SimulationError(f"{operand.name} read but not held")
        return self.registers[operand.name]

    def run(self, body) -> None:
        regions: List[bool] = []  # for each open region, whether it is active (its control holds)
        for instruction in body:
            self.steps += 1
            if self.steps > MAX_STEPS:
                raise SimulationError("step limit reached")
            if isinstance(instruction, (Ctrl, ICtrl)):
                regions.append(bool(self.read(instruction.control)) == isinstance(instruction, Ctrl))
            elif isinstance(instruction, EndCtrl):
                if not regions:
                    raise SimulationError("ENDCTRL with no open region")
                regions.pop()
            elif regions and isinstance(instruction, (Alloc, Free)):
                raise SimulationError("a controlled region may not allocate or free")
            elif all(regions):  # inside a region whose control does not hold, an instruction has no effect
                self.execute(instruction)
        if regions:
            raise SimulationError("a controlled region is never closed")

    def execute(self, instruction) -> None:
        if isinstance(instruction, Alloc):
            self.alloc(instruction.register)
        elif isinstance(instruction, Free):
            self.free(instruction.register)
        elif isinstance(instruction, (Op, IOp)):
            self.operate(instruction)
        elif isinstance(instruction, Update):
            self.update(instruction)

    def alloc(self, name: str) -> None:
        if name in self.registers:
            raise SimulationError(f"{name} allocated twice")
        self.registers[name] = 0
        self.peak = max(self.peak, len(self.registers))

    def free(self, name: str) -> None:
        if name not in self.registers:
            raise SimulationError(f"{name} freed but not held")
        if self.registers[name] != 0:
            raise SimulationError(f"{name} freed while holding {self.registers[name]}")
        del self.registers[name]

    def update(self, instruction: Update) -> None:
        name = instruction.dst
        if name not in self.registers:
            raise SimulationError(f"{name} updated but not allocated")
        value = _OPCODES[instruction.opcode](self.registers[name], self.read(instruction.source))
        self.registers[name] = self.norm(value)

    def operate(self, instruction) -> None:
        value = _OPCODES[instruction.opcode](*(self.read(source) for source in instruction.sources))
        if instruction.opcode in _ARITHMETIC:
            value = self.norm(value)
        name = instruction.dst
        if name not in self.registers:
            raise SimulationError(f"{name} written but not allocated")
        held = self.registers[name]
        if isinstance(instruction, Op):  # computes into a register that is zero
            if held != 0:
                raise SimulationError(f"{instruction.opcode.upper()} writes {name}, which holds {held}, not zero")
            self.registers[name] = value
        else:  # undoes it: the register must hold exactly what the instruction computes
            if held != value:
                raise SimulationError(f"I{instruction.opcode.upper()} expected {name} to hold {value} but it holds {held}")
            self.registers[name] = 0


def simulate_isa(program: IsaProgram, system: Optional[System] = None) -> Result:
    """Run a program; returns its value, the registers left held and the most held at once."""
    machine = _IsaMachine(system)
    machine.run(program.body)
    value = None if program.result is None else machine.read(program.result)
    held = set(machine.registers) - ({program.result.name} if isinstance(program.result, Reg) else set())
    return Result(value, {"<program>": held}, {"<program>": machine.peak})


def execute(output: Any, system: Optional[System] = None) -> Optional[Result]:
    """
    Run what a pipeline stage produced on its reference machine: a program (a list of nodes) on the program machine,
    instructions on the instruction machine. A circuit has no machine here, so it gives None. Give the system the
    program was compiled for to wrap integers to its register width.
    """
    if isinstance(output, list):
        return simulate(output, system)
    if isinstance(output, IsaProgram):
        return simulate_isa(output, system)
    return None

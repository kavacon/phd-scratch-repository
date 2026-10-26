"""
Condition bits, in the source after every other change to the conditionals: a conditional is on a dedicated boolean variable.

A region is controlled on one qubit, so the register it is controlled on has to be 0 or 1 there. A condition that holds
any other number has a nonzero value that is true and a qubit that says nothing about it, so the conditional would run
differently in the circuit than the source says. By now each conditional has its guard, as a variable, so a guard that is
not a comparison or a logical operation is raised into a variable of its own, `$b = c != 0`, just before the conditional,
and the conditional is on that. A guard that already is a boolean is used as it is, unless the conditional also reads it:
an adder changes the bits of the operand it reads for a while, which would change the qubit the region is controlled on,
so that guard gets a variable of its own too (`$b = c && true`, one gate). The variable is shared by the conditionals that
follow it on the same guard until the guard is assigned again. A guard that nothing inside the outermost conditional assigns
is raised before that conditional, so a conditional inside another is still controlled on a bit made outside every region.
"""
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Set

from compiler.compiler import NODE, compile_pass
from compiler.parser.ast_tools import walk
from compiler.parser.grammar import Assign, BinOp, Boolean, Equals, If, Node, Not, Number, Variable
from compiler.passes.names import assigned, read
from compiler.system import System

BOOLEAN_OPERATORS = {"<", ">", "!=", "&&", "||"}


def _boolean_names(program: Sequence[Node]) -> Set[str]:
    """Variables that are only ever assigned a comparison, a logical operation or a boolean."""
    assignments: Dict[str, List[Node]] = {}
    for statement in program:
        for node in walk(statement):
            if isinstance(node, Assign):
                assignments.setdefault(node.name, []).append(node.value)
    names: Set[str] = set()
    changed = True
    while changed:  # a copy of a boolean is a boolean
        changed = False
        for name, values in assignments.items():
            if name not in names and all(_boolean(v, names) for v in values):
                names.add(name)
                changed = True
    return names


def _boolean(value: Node, names: Set[str]) -> bool:
    if isinstance(value, BinOp):
        return value.op in BOOLEAN_OPERATORS
    return isinstance(value, (Not, Equals, Boolean)) or (isinstance(value, Variable) and value.name in names)


class _Raiser:
    def __init__(self, booleans: Set[str]) -> None:
        self.booleans = booleans
        self.count = 0

    def top(self, program: Sequence[Node]) -> List[Node]:
        out: List[Node] = []
        bits: Dict[str, str] = {}  # a guard -> the variable that holds whether it is true, while the guard is unchanged
        for statement in program:
            if isinstance(statement, If):
                changed = set(assigned([statement]))
                local, rewritten = self.conditional(statement, changed, bits, out)
                out.extend(local)
                for name in changed:
                    bits.pop(name, None)
                statement = rewritten
            elif isinstance(statement, Assign):
                bits.pop(statement.name, None)
            out.append(statement)
        return out

    def conditional(self, statement: If, changed: Set[str], bits: Dict[str, str], before: List[Node]):
        """
        The conditional with its guard raised, and the assignments to put just before it. A guard that nothing inside the
        outermost conditional assigns is raised before that one, so a conditional inside another is still controlled on a
        bit made outside every region (`before` is where those go); any other guard is raised in its own block.
        """
        local: List[Node] = []
        condition = statement.condition
        if isinstance(condition, Variable):
            guard = condition.name
            inside = set().union(*map(read, statement.body + statement.orelse))
            if guard not in self.booleans or guard in inside:
                bit_for, put = (bits, before) if guard not in changed else ({}, local)
                if guard not in bit_for:
                    self.count += 1
                    bit_for[guard] = f"$b{self.count}"
                    value = (BinOp("&&", Variable(guard), Boolean(True)) if guard in self.booleans
                             else BinOp("!=", Variable(guard), Number(0)))
                    put.append(Assign(bit_for[guard], value))
                condition = Variable(bit_for[guard])
        body = self.nested(statement.body, changed, bits, before)
        orelse = self.nested(statement.orelse, changed, bits, before)
        return local, replace(statement, condition=condition, body=body, orelse=orelse)

    def nested(self, block: Sequence[Node], changed: Set[str], bits: Dict[str, str], before: List[Node]):
        out: List[Node] = []
        for statement in block:
            if isinstance(statement, If):
                local, statement = self.conditional(statement, changed, bits, before)
                out.extend(local)
            out.append(statement)
        return tuple(out)


@compile_pass(
    NODE, NODE,
    title="Condition bits",
    requires=("balance",),
    summary="Puts the guard of each conditional in a boolean variable of its own where it is not already a comparison "
            "or logical operation, or the conditional reads it.",
)
def condition_bits(program: List[Node], system: Optional[System] = None) -> List[Node]:
    return _Raiser(_boolean_names(program)).top(program)

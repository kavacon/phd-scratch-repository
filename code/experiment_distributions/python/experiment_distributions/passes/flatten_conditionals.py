"""
Flatten nested conditionals in the source, before balancing, by combining the conditions.

    if a { if b { X } else { Y } }
becomes, when X assigns every outer variable Y does and does not read what Y assigns,
    $c = a && b;   if a { Y }   if $c { X }
where the second conditional overwrites what the first wrote where both hold, so no `a && !b` is needed; and when Y is
empty the first conditional is empty and goes, leaving the single conditional `if $c { X }`. When X does not overwrite
everything Y writes, or reads something it writes, Y keeps its own condition, `$d = a && !b`. A conditional with an else
arm is split into one on the condition and one on its negation first, and an else arm that is only a conditional is
treated the same way. The results are flattened again until nothing is nested.

Only a conditional whose then-arm is a single conditional is flattened, apart from assignments to compiler temporaries
that compute its condition, which are moved out in front (they have no effect, and run harmlessly when the outer
condition does not hold). Anything else is left nested, since a name first assigned in one arm is not visible in another.
It runs before balancing so each conditional that is left only gets identity arms for what it assigns.
"""
from typing import List, Optional, Set

from compiler.compiler import NODE, compile_pass
from compiler.parser.grammar import Assign, BinOp, If, Node, Not, Variable
from compiler.passes.names import assigned, read, variables
from compiler.system import System


def _identity(statement: Node) -> bool:
    return isinstance(statement, Assign) and isinstance(statement.value, Variable) and statement.value.name == statement.name


def _empty(block) -> bool:
    return all(_identity(s) for s in block)


class _Flattener:
    def __init__(self) -> None:
        self.count = 0

    def fresh(self, prefix: str) -> str:
        self.count += 1
        return f"${prefix}{self.count}"

    def run(self, statement: If, before: Set[str]) -> List[Node]:
        condition = statement.condition
        if not isinstance(condition, Variable):
            return [statement]
        nested = self.split(statement.body, before)
        else_nested = len(statement.orelse) == 1 and isinstance(statement.orelse[0], If)
        if nested is None and not else_nested:
            return [statement]
        if statement.orelse:  # one conditional on the condition and one on its negation
            negation = self.fresh("n")
            first = self.run(If(condition, statement.body, ()), before)
            second = self.run(If(Variable(negation), statement.orelse, ()), before)
            return [Assign(negation, Not(condition))] + first + second
        prefix, inner = nested
        a, b = condition.name, inner.condition.name
        conjunction = self.fresh("c")
        out: List[Node] = prefix + [Assign(conjunction, BinOp("&&", Variable(a), Variable(b)))]
        x, y = inner.body, inner.orelse
        known = before | {p.name for p in prefix}
        if _empty(y):
            return out + self.run(If(Variable(conjunction), x, ()), known)
        wrote_x, wrote_y = set(assigned(x)) & known, set(assigned(y)) & known
        if wrote_y <= wrote_x and not set().union(*map(read, x)) & set(assigned(y)):
            return out + self.run(If(condition, y, ()), known) + self.run(If(Variable(conjunction), x, ()), known)
        negation, other = self.fresh("n"), self.fresh("d")
        out += [Assign(negation, Not(Variable(b))), Assign(other, BinOp("&&", Variable(a), Variable(negation)))]
        return out + self.run(If(Variable(conjunction), x, ()), known) + self.run(If(Variable(other), y, ()), known)

    def split(self, body, before: Set[str]):
        """The assignments to condition temporaries to move out and the one conditional that makes up the rest."""
        prefix, rest = [], list(body)
        while (rest and isinstance(rest[0], Assign) and rest[0].name.startswith("$") and not rest[0].inplace
               and variables(rest[0].value) <= before):
            prefix.append(rest.pop(0))
        if len(rest) != 1 or not isinstance(rest[0], If) or not isinstance(rest[0].condition, Variable):
            return None
        if rest[0].condition.name not in before | {p.name for p in prefix}:
            return None
        return prefix, rest[0]


@compile_pass(
    NODE, NODE,
    title="Flatten nested conditionals",
    requires=("flatten",),
    summary="Combines the conditions of nested conditionals into one conjunction, so a conditional inside a conditional "
            "becomes one at the same level, and drops the arm that ends up empty.",
)
def flatten_conditionals(program: List[Node], system: Optional[System] = None) -> List[Node]:
    flattener, out = _Flattener(), []
    for statement in program:
        if isinstance(statement, If):
            out.extend(flattener.run(statement, set(assigned(out))))
        else:
            out.append(statement)
    return out

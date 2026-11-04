"""
Translate a source program (the typed AST from `compiler.parser`) into MLIR text: the standard `func`, `arith`, `scf`,
`affine` and `memref` dialects for definite values, and the `dist` dialect for distributions, which the C++ tool in
`cpp/` reads.

The translation is mechanical and keeps the source's mutable variables: each variable is a slot (`memref.alloca`) that is
loaded where it is read and stored where it is assigned, and `mem2reg` turns the slots into SSA values. Every integer has
one fixed width (`WIDTH`), an upper bound that range analysis narrows later.

A variable is either plain or a reference. Which one is decided by name, from the syntax alone (`_may_hold_distribution`):
a variable that is assigned or updated with something that involves a distribution, or inside a conditional whose condition
does, may hold a distribution on some path. Its slot holds a reference to an object, a cell (`memref<!dist.dist<i32>>`),
read and written with `memref.load` and `memref.store`; every other variable is a plain integer.
What `y = x` means (the semantics: a copy of an integer, the same object for a distribution) depends on what the object
holds at that point, which only folding shows. So the translation emits both and selects: `arith.select` of a copy and of
x on `dist.is_single` of x's value, which folds once that is known. The same question guards `y = e` over a name that holds
a distribution (a `cf.assert`: the semantics refuse to drop one).
- a conditional is an `scf.if` on the condition's `dist.cond` (an integer when the condition is definite). Both arms are
  ordinary regions: `resolve-dist` keeps the arm a definite condition takes, and turns one on a superposition into both
  arms and a `dist.select` for each cell either arm changed;
- a computed distribution assigned to a new variable is marked with `dist.instance` (see `resolve-instances`);
- an update is computed out of place and stored back into the same cell; `in-place-updates` finds the ones that
  can be done in the target's register.
A loop with literal bounds is an `affine.for`, unrolled before the slots are promoted. A call is a `func.call` of a function
translated once for each combination of argument types (an integer, or a reference), removed by the stock `inline` pass. A
distribution argument is `share`d, so it is the caller's object when it holds a superposition and a copy when it does not;
a function that returns a variable returns it the same way.
"""
import itertools
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple

from compiler.parser.grammar import (Assign, BinOp, Boolean, Call, Distribution, Equals, ExprStmt, ForLoop, FunctionDef, If,
                                     Node, Not, Number, Return, Variable)

WIDTH = 32
MAX_DEPTH = 8  # loops nested in one another
DROP = "assigning to a name would drop the distribution it holds; only an in-place update changes one"
MAX_ITERATIONS = 10_000  # times the innermost body runs, which is how many copies unrolling makes


class TranslationError(ValueError):
    pass


class _Value(NamedTuple):
    ssa: str
    type: str  # "i32", "i1", "!dist.dist<i32>", "!dist.dist<i1>" or a cell

    @property
    def dist(self) -> bool:
        return self.type.startswith("!dist.dist")


class _Var(NamedTuple):
    slot: str  # the `memref.alloca` that holds the variable
    ref: bool  # whether it holds a reference to an object rather than an integer
    type: str  # what the slot holds: an integer type (a comparison's `i1` too) or a reference


def _mentions(node: Node, names: Set[str], functions: Dict[str, FunctionDef]) -> bool:
    """Whether an expression involves a distribution: a range, a name that may hold one, or a call that may make one."""
    if isinstance(node, Distribution):
        return node.stop.value - node.start.value > 1
    if isinstance(node, Variable):
        return node.name in names
    if isinstance(node, BinOp):
        return _mentions(node.left, names, functions) or _mentions(node.right, names, functions)
    if isinstance(node, Not):
        return _mentions(node.operand, names, functions)
    if isinstance(node, Equals):
        return any(_mentions(operand, names, functions) for operand in node.operands)
    if isinstance(node, Call):
        return any(_mentions(argument, names, functions) for argument in node.args) or _makes_range(node.name, functions, [])
    return False


def _makes_range(name: str, functions: Dict[str, FunctionDef], active: List[str]) -> bool:
    """Whether a function (or one it calls) contains a range, so its result may be a distribution whatever it is given."""
    if name not in functions or name in active:
        return False
    active.append(name)
    found = any((isinstance(node, Distribution) and node.stop.value - node.start.value > 1)
                or (isinstance(node, Call) and _makes_range(node.name, functions, active))
                for node in _walk(functions[name].body))
    active.pop()
    return found


def _walk(statements: Sequence[Node]) -> Iterator[Node]:
    """Every expression and statement under these statements."""
    for node in statements:
        yield node
        if isinstance(node, Assign):
            yield from _expressions(node.value)
        elif isinstance(node, If):
            yield from _expressions(node.condition)
            yield from _walk([*node.body, *node.orelse])
        elif isinstance(node, ForLoop):
            yield from _walk(node.body)
        elif isinstance(node, (ExprStmt, Return)):
            yield from _expressions(node.expr if isinstance(node, ExprStmt) else node.value)


def _expressions(node: Node) -> Iterator[Node]:
    yield node
    if isinstance(node, BinOp):
        yield from _expressions(node.left)
        yield from _expressions(node.right)
    elif isinstance(node, Not):
        yield from _expressions(node.operand)
    elif isinstance(node, Equals):
        for operand in node.operands:
            yield from _expressions(operand)
    elif isinstance(node, Call):
        for argument in node.args:
            yield from _expressions(argument)


def _may_hold_distribution(body: Sequence[Node], seeds: Set[str], functions: Dict[str, FunctionDef]) -> Set[str]:
    """
    The names that may hold a distribution on some path, from the syntax alone: the parameters given a reference, and any
    name assigned or updated with an expression that involves one, or anywhere inside a conditional whose condition does
    (the conditional then ties the name to its condition). Grown until nothing is added. A name that is too many is only a
    reference where an integer would do, which costs nothing once the object is resolved.
    """
    names = set(seeds)

    def visit(statements: Sequence[Node], tied: bool) -> None:
        for node in statements:
            if isinstance(node, Assign) and (tied or _mentions(node.value, names, functions)):
                names.add(node.name)
            elif isinstance(node, If):
                inner = tied or _mentions(node.condition, names, functions)
                visit([*node.body, *node.orelse], inner)
            elif isinstance(node, ForLoop):
                visit(node.body, tied)

    while True:
        before = len(names)
        visit(body, False)
        if len(names) == before:
            return names


class _Instance(NamedTuple):
    """One translation of a function, for the kinds of argument it was called with."""
    symbol: str
    result: str  # the result type in the IR
    kind: str  # "ref": returns a variable it holds, as a reference; "fresh": a new range; "value": any other value


class _Module:
    """
    The functions of a program. A function has no declared types, so it is translated once for each combination of
    argument types it is called with: an integer is passed as a value, which makes the parameter a copy, and a distribution
    as a reference (which the caller `share`d, so it is the caller's own object or a copy of an integer).
    """

    def __init__(self, program: Sequence[Node], width: int) -> None:
        self.width = width
        self.functions = {node.name: node for node in program if isinstance(node, FunctionDef)}
        self.instances: Dict[tuple, _Instance] = {}
        self.texts: List[str] = []
        self.active: List[str] = []

    def instance(self, definition: FunctionDef, kinds: Sequence[str]) -> _Instance:
        key = (definition.name, tuple(kinds))
        if key in self.instances:
            return self.instances[key]
        if definition.name in self.active:
            raise TranslationError(f"{definition.name} calls itself, which a circuit cannot do")
        self.active.append(definition.name)
        earlier = sum(1 for other in self.instances if other[0] == definition.name)
        symbol = definition.name if earlier == 0 else f"{definition.name}_{earlier}"
        *statements, final = definition.body
        if not isinstance(final, Return):
            raise TranslationError(f"{definition.name} must end with a return")
        references = {p.name for p, kind in zip(definition.params, kinds) if kind.startswith("memref<")}
        body = _Translator(self.width, self, _may_hold_distribution(definition.body, references, self.functions))
        arguments = []
        for parameter, kind in zip(definition.params, kinds):
            argument = f"%a{len(arguments)}"
            arguments.append(f"{argument}: {kind}")
            if kind.startswith("memref<"):  # the caller's object (or a copy of its integer), held as it is
                body.declare(parameter.name, _Value(argument, kind))
            else:
                body.bind(parameter.name, _Value(argument, kind))
        for statement in statements:
            body.statement(statement)
        value, result_kind = body.returned(final.value)
        lines = ["  " + line for line in [*body.lines, f"func.return {value.ssa} : {value.type}"]]
        self.texts.append("\n".join([f"func.func private @{symbol}({', '.join(arguments)}) -> {value.type} {{", *lines, "}"]))
        self.active.pop()
        self.instances[key] = _Instance(symbol, value.type, result_kind)
        return self.instances[key]


class _Translator:
    """
    Translates the statements of one function body, or of the program. It keeps one thing besides the text: the names in
    scope, each bound to its slot. A name first assigned in a branch or a loop body ends with it. Nothing here says what two
    names mean for one object; that is in the values the slots hold.
    """

    def __init__(self, width: int, module: _Module, distribution_names: Set[str]) -> None:
        self.module = module
        self.width = width
        self.type = f"i{width}"
        self.names = itertools.count()
        self.lines: List[str] = []
        self.level = 0
        self.scopes: List[Dict[str, _Var]] = [{}]
        self.distribution_names = distribution_names
        self.loops: List[str] = []  # the variables of the loops this is inside, which the body may only read
        self.iterations = 1  # how often the statements run, from the loops they are inside

    # -- text ------------------------------------------------------------------------------------------------------

    def out(self, text: str) -> None:
        self.lines.append("  " * self.level + text)

    def emit(self, text: str, type: str) -> _Value:
        name = f"%{next(self.names)}"
        self.out(f"{name} = {text}")
        return _Value(name, type)

    def constant(self, value: int) -> _Value:
        low, high = -(1 << (self.width - 1)), (1 << (self.width - 1)) - 1
        if not low <= value <= high:
            raise TranslationError(f"{value} does not fit in {self.type}")
        return self.emit(f"arith.constant {value} : {self.type}", self.type)

    def lift(self, value: _Value) -> _Value:
        if value.dist:
            return value
        return self.emit(f"dist.lift {value.ssa} : {value.type} -> !dist.dist<{value.type}>", f"!dist.dist<{value.type}>")

    # -- variables -------------------------------------------------------------------------------------------------

    def lookup(self, name: str) -> _Var:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        raise TranslationError(f"{name} is used before it is assigned")

    def exists(self, name: str) -> bool:
        return any(name in scope for scope in self.scopes)

    def slot(self, name: str, type: str) -> _Var:
        ssa = f"%{next(self.names)}"
        self.out(f"{ssa} = memref.alloca() : memref<{type}>")
        self.scopes[-1][name] = _Var(ssa, type.startswith("memref<"), type)
        return self.scopes[-1][name]

    def store(self, var: _Var, value: _Value) -> None:
        self.out(f"memref.store {value.ssa}, {var.slot}[] : memref<{value.type}>")

    def reference(self, var: _Var) -> _Value:
        return self.emit(f"memref.load {var.slot}[] : memref<{var.type}>", var.type)

    def declare(self, name: str, reference: _Value) -> None:
        """Bind a name to a reference."""
        self.store(self.slot(name, reference.type), reference)

    def bind(self, name: str, value: _Value) -> None:
        """A new name for a value: an integer in a slot of its own, or a new object when the name may hold a distribution."""
        if name in self.distribution_names:
            self.declare(name, self.new(self.lift(value)))
        elif value.dist:
            raise TranslationError(f"{name} holds a distribution but is not known to")
        else:
            self.store(self.slot(name, value.type), value)

    def read(self, name: str) -> _Value:
        var = self.lookup(name)
        if var.ref:
            return self.get(self.reference(var))
        return self.emit(f"memref.load {var.slot}[] : memref<{var.type}>", var.type)

    def get(self, reference: _Value) -> _Value:
        return self.emit(f"memref.load {reference.ssa}[] : {reference.type}", reference.type[len("memref<"):-1])

    def new(self, value: _Value) -> _Value:
        """A new object holding a distribution."""
        ref = f"memref<{value.type}>"
        cell = self.emit(f"memref.alloc() : {ref}", ref)  # (alloc, not alloca: stock mem2reg must not promote a cell before the conditionals on a superposition are resolved)
        self.out(f"memref.store {value.ssa}, {cell.ssa}[] : {cell.type}")
        return cell

    def single(self, value: _Value) -> _Value:
        return self.emit(f"dist.is_single {value.ssa} : {value.type} -> i1", "i1")

    def share(self, source: _Value) -> _Value:
        """The object a new name for `source` means: a copy if it holds one value, the object itself otherwise."""
        held = self.get(source)
        copy = self.new(held)
        single = self.single(held)
        return self.emit(f"arith.select {single.ssa}, {copy.ssa}, {source.ssa} : {source.type}", source.type)

    def check_single(self, reference: _Value, message: str) -> _Value:
        """The distribution an object holds, which must be one value (or the program is refused with `message`)."""
        held = self.get(reference)
        self.out(f'cf.assert {self.single(held).ssa}, "{message}"')
        return held

    def check_writable(self, name: str) -> None:
        if name in self.loops:
            raise TranslationError(f"the loop variable {name} cannot be assigned to inside its loop")

    # -- expressions -----------------------------------------------------------------------------------------------

    def operation(self, name: str, left: _Value, right: _Value, boolean: bool = False) -> _Value:
        """`name` (arith or dist, by whether an operand is a distribution) on two values of one type. A comparison, or a
        boolean operation, gives one-bit integers."""
        one = "i1" if boolean else self.type
        if left.dist or right.dist:
            result = f"!dist.dist<{one}>"
            return self.emit(f"dist.{name} {left.ssa}, {right.ssa} : {left.type}, {right.type} -> {result}", result)
        if name in ("slt", "eq"):
            return self.emit(f"arith.cmpi {name}, {left.ssa}, {right.ssa} : {self.type}", "i1")
        return self.emit(f"arith.{name} {left.ssa}, {right.ssa} : {left.type}", left.type)

    def flag(self, value: _Value) -> _Value:
        """A value as a boolean: a one-bit integer already is one, and any other is true when it is not zero."""
        if value.type in ("i1", "!dist.dist<i1>"):
            return value
        return self.negate(self.operation("eq", value, self.constant(0), True))

    def negate(self, flag: _Value) -> _Value:
        return self.operation("xori", flag, self.emit("arith.constant true", "i1"), True)

    def binary(self, node: BinOp) -> _Value:
        left, right = self.expression(node.left), self.expression(node.right)
        arithmetic = {"+": "addi", "-": "subi", "*": "muli"}
        if node.op in arithmetic:
            return self.operation(arithmetic[node.op], left, right)
        if node.op == "<":
            return self.operation("slt", left, right, True)
        if node.op == ">":
            return self.operation("slt", right, left, True)
        if node.op == "!=":
            return self.negate(self.operation("eq", left, right, True))
        if node.op in ("&&", "||"):
            return self.operation("andi" if node.op == "&&" else "ori", self.flag(left), self.flag(right), True)
        raise TranslationError(f"operator {node.op!r} is not supported yet")

    def equals(self, node: Equals) -> _Value:
        """`a == b == c` holds when every neighbouring pair is equal."""
        values = [self.expression(operand) for operand in node.operands]
        result = self.operation("eq", values[0], values[1], True)
        for left, right in zip(values[1:], values[2:]):
            result = self.operation("andi", result, self.operation("eq", left, right, True), True)
        return result

    def expression(self, node: Node) -> _Value:
        if isinstance(node, Number):
            if not isinstance(node.value, int):
                raise TranslationError("only integer constants are supported yet")
            return self.constant(node.value)
        if isinstance(node, Variable):
            return self.read(node.name)
        if isinstance(node, BinOp):
            return self.binary(node)
        if isinstance(node, Equals):
            return self.equals(node)
        if isinstance(node, Not):
            return self.negate(self.flag(self.expression(node.operand)))
        if isinstance(node, Boolean):
            return self.emit(f"arith.constant {'true' if node.value else 'false'}", "i1")
        if isinstance(node, Call):
            value, _ = self.call(node)
            return self.get(value) if value.type.startswith("memref<") else value
        if isinstance(node, Distribution):
            start, stop = node.start.value, node.stop.value
            if stop <= start:
                raise TranslationError(f"the range {start}..{stop} is empty")
            if stop - start == 1:  # a range one value wide is an ordinary definite value
                return self.constant(start)
            return self.emit(f"dist.uniform {start}, {stop} : !dist.dist<{self.type}>", f"!dist.dist<{self.type}>")
        raise TranslationError(f"{type(node).__name__} is not supported yet")

    def object_of(self, node: Node) -> Optional[_Value]:
        """The reference a plain name stands for, when the name may hold a distribution."""
        if isinstance(node, Variable) and self.exists(node.name) and self.lookup(node.name).ref:
            return self.reference(self.lookup(node.name))
        return None

    def instance_of(self, node: Node, value: _Value) -> _Value:
        """A computed distribution is an independent instance; a range is already a new one."""
        if value.dist and not isinstance(node, Distribution):
            return self.emit(f"dist.instance {value.ssa} : {value.type} -> {value.type}", value.type)
        return value

    def call(self, node: Call) -> Tuple[_Value, _Instance]:
        """
        A call passes an integer as a value and a distribution as a reference: a plain name that may hold one is `share`d
        (the caller's object if it holds a superposition, a copy if it holds an integer), and a computed distribution is
        made an independent instance in an object of its own.
        """
        definition = self.module.functions.get(node.name)
        if definition is None:
            raise TranslationError(f"{node.name} is not a function")
        if len(node.args) != len(definition.params):
            raise TranslationError(f"{node.name} takes {len(definition.params)} argument(s) but is called with "
                                   f"{len(node.args)}")
        passed = []
        for argument in node.args:
            reference = self.object_of(argument)
            if reference is not None:
                passed.append(self.share(reference))
                continue
            value = self.expression(argument)
            if value.dist:
                value = self.instance_of(argument, value)
                value = self.new(value)
            passed.append(value)
        instance = self.module.instance(definition, [value.type for value in passed])
        arguments = ", ".join(value.ssa for value in passed)
        text = f"func.call @{instance.symbol}({arguments}) : ({', '.join(value.type for value in passed)}) -> {instance.result}"
        return self.emit(text, instance.result), instance

    def returned(self, node: Node) -> Tuple[_Value, str]:
        """What a function hands back: a variable it holds (as a reference, shared), or a value."""
        reference = self.object_of(node)
        if reference is not None:
            return self.share(reference), "ref"
        if isinstance(node, Call):
            value, instance = self.call(node)
            return value, instance.kind
        return self.expression(node), "fresh" if isinstance(node, Distribution) else "value"

    # -- statements ------------------------------------------------------------------------------------------------

    def region(self, statements: Sequence[Node]) -> None:
        """Translate statements in a region of their own: names first assigned in them end with them."""
        self.scopes.append({})
        self.level += 1
        for statement in statements:
            self.statement(statement)
        self.level -= 1
        self.scopes.pop()

    def conditional(self, node: If) -> None:
        condition = self.flag(self.expression(node.condition))
        if condition.dist:  # an integer if the distribution turns out to be a lifted one: the folding decides
            condition = self.emit(f"dist.cond {condition.ssa} : {condition.type} -> i1", "i1")
        self.out(f"scf.if {condition.ssa} {{")
        self.region(node.body)
        self.out("} else {")
        self.region(node.orelse)
        self.out("}")

    def loop(self, node: ForLoop) -> None:
        """A loop with literal bounds is an `affine.for`, whose variable the body may only read. One that runs no times is
        translated for its errors and then dropped."""
        start, stop = node.start.value, node.stop.value
        iterations = self.iterations * max(stop - start, 1)
        if len(self.loops) + 1 > MAX_DEPTH or iterations > MAX_ITERATIONS:
            raise TranslationError(f"loops nested {MAX_DEPTH} deep or running {MAX_ITERATIONS} iterations in all cannot "
                                   "be unrolled")
        mark = len(self.lines)
        index = f"%{next(self.names)}"
        self.out(f"affine.for {index} = {start} to {stop} {{")
        self.scopes.append({})
        self.level += 1
        self.loops.append(node.var)
        self.iterations = iterations
        counter = self.emit(f"arith.index_cast {index} : index to {self.type}", self.type)
        self.bind(node.var, counter)
        for statement in node.body:
            self.statement(statement)
        self.iterations //= max(stop - start, 1)
        self.loops.pop()
        self.level -= 1
        self.scopes.pop()
        self.out("}")
        if stop <= start:
            del self.lines[mark:]

    def update(self, node: Assign) -> None:
        """`name += amount;` or `name -= amount;`, which the parser gives as `name = name + amount` marked in place. For a
        distribution the sum is computed out of place from what the object holds and stored back into it."""
        var = self.lookup(node.name)
        self.check_writable(node.name)
        name = {"+": "add", "-": "sub"}[node.value.op]
        if var.ref:
            reference = self.reference(var)
            held = self.get(reference)
            amount = self.expression(node.value.right)
            result = self.emit(f"dist.{name}i {held.ssa}, {amount.ssa} : {held.type}, {amount.type} -> {held.type}", held.type)
            self.out(f"memref.store {result.ssa}, {reference.ssa}[] : {reference.type}")
            return
        target = self.read(node.name)
        amount = self.expression(node.value.right)
        if amount.dist:
            raise TranslationError(f"{node.name} is updated with a distribution but is not known to hold one")
        self.store(var, self.emit(f"arith.{name}i {target.ssa}, {amount.ssa} : {self.type}", self.type))

    def assign(self, node: Assign) -> None:
        exists = self.exists(node.name)
        if exists:
            self.check_writable(node.name)
        source = self.object_of(node.value)
        fresh = isinstance(node.value, Distribution)
        if source is None and isinstance(node.value, Call):
            result, instance = self.call(node.value)
            if result.type.startswith("memref<"):
                source = result
            else:
                value, fresh = result, instance.kind == "fresh"
        elif source is None:
            value = self.expression(node.value)
        if source is not None:  # a plain name, or a variable a call returned: a copy or an alias, once it is known
            if not exists:
                self.declare(node.name, self.share(source))
                return
            var = self.lookup(node.name)
            if not var.ref:
                raise TranslationError(f"{node.name} takes a distribution but is not known to hold one")
            target = self.reference(var)
            held = self.check_single(target, DROP)
            incoming = self.get(source)
            single = self.single(incoming)
            copied = self.emit(f"arith.select {single.ssa}, {incoming.ssa}, {held.ssa} : {held.type}", held.type)
            self.out(f"memref.store {copied.ssa}, {target.ssa}[] : {target.type}")
            self.store(var, self.emit(f"arith.select {single.ssa}, {target.ssa}, {source.ssa} : {target.type}", target.type))
            return
        if value.dist and not fresh:
            value = self.instance_of(node.value, value)
        if not exists:
            self.bind(node.name, value)
            return
        var = self.lookup(node.name)
        if var.ref:
            target = self.reference(var)
            self.check_single(target, DROP)
            self.out(f"memref.store {self.lift(value).ssa}, {target.ssa}[] : {target.type}")
        elif value.dist:
            raise TranslationError(f"{node.name} takes a distribution but is not known to hold one")
        else:
            self.store(var, value)

    def statement(self, node: Node) -> None:
        if isinstance(node, Assign) and node.inplace:
            self.update(node)
        elif isinstance(node, Assign):
            self.assign(node)
        elif isinstance(node, If):
            self.conditional(node)
        elif isinstance(node, ForLoop):
            self.loop(node)
        elif isinstance(node, ExprStmt):  # the value is not used, but a call still runs
            self.expression(node.expr)
        else:
            raise TranslationError(f"{type(node).__name__} is not supported yet")

    def program(self, program: Sequence[Node]) -> str:
        returned = None
        for statement in program:
            if isinstance(statement, Return):
                returned = self.expression(statement.value)
            else:
                self.statement(statement)
        if returned is None:
            raise TranslationError("a program must end with a return")
        if returned.dist:  # using a distribution as the result measures it
            element = returned.type[len("!dist.dist<"):-1]
            returned = self.emit(f"dist.measure {returned.ssa} : {returned.type} -> {element}", element)
        if returned.type == "i1":  # the result of a comparison or a boolean
            returned = self.emit(f"arith.extui {returned.ssa} : i1 to {self.type}", self.type)
        body = ["  " + line for line in [*self.lines, f"func.return {returned.ssa} : {self.type}"]]
        return "\n".join([f"func.func @main() -> {self.type} {{", *body, "}"])


def translate(program: Sequence[Node], width: int = WIDTH) -> str:
    module = _Module(program, width)
    statements = [node for node in program if not isinstance(node, FunctionDef)]
    main = _Translator(width, module, _may_hold_distribution(statements, set(), module.functions))
    text = main.program(statements)
    return "\n\n".join([*module.texts, text])

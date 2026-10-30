"""
Translate a source program (the typed AST from `compiler.parser`) into MLIR text: the standard `func`, `arith`, `scf`,
`affine` and `memref` dialects for definite values, and the `dist` dialect for distributions, which the C++ tool in
`cpp/` reads.

The translation is mechanical and keeps the source's mutable variables: each variable is a slot (`memref.alloca`) that is
loaded where it is read and stored where it is assigned. The pass `mem2reg` then turns the slots into SSA values, which is
where every analysis and the lowering start. Reason: a name keeps meaning "the variable as it is now" however an update
reaches it (a loop, a branch, a call), and the pass finds the values, so the translation tracks no versions.

Every integer is given one fixed width, chosen by the front-end, the way a classical front-end gives every `int` its
declared width. That width is an upper bound: range analysis narrows it later. Results that do not fit wrap, as they do
in C. Change `WIDTH` (or pass `width`) to try another bound.

A variable is a distribution if it was made by a range (`x = 0..4;`) or computed from one. What assignment means for
one (see the decision "A distribution denotes a law"):
- a plain name on the right (`b = a;`) is an alias: both names are the same slot, so an in-place update through either
  is seen through both. Definite values are copied, as integers are;
- a computed value (an expression involving a distribution) assigned to a new variable is marked with `dist.instance`,
  an independent distribution with the law of the expression. The translation only marks it: a pass, `resolve-instances`,
  decides how the instance is made;
- within one expression a distribution is one roll, so `a - a` is zero;
- `+=` and `-=` update the variable in place, including with another distribution;
- overwriting a distribution by a plain assignment would drop it and is refused.
A conditional on a distribution computes both arms on copies of the variables they change and picks each with one
`dist.select`; one on a definite value is an `scf.if` whose arms store to the variables. A loop has literal bounds, so
it is an `affine.for`, which a pass unrolls before the slots are promoted. A call is its body inlined with each parameter
assigned from its argument (see `_Module`): a function is a `func.func` per combination of argument types, removed by the
stock `inline` pass, and a function that returns a distribution it holds returns the slot, so the result is an alias.
Anything else is refused by name, so a later milestone adds it deliberately.
"""
import itertools
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple, Union

from compiler.parser.grammar import (Assign, BinOp, Call, Distribution, ExprStmt, ForLoop, FunctionDef, If, Node, Number,
                                     Return, Variable)

WIDTH = 32
MAX_DEPTH = 8  # loops nested in one another
MAX_ITERATIONS = 10_000  # times the innermost body runs, which is how many copies unrolling makes


class TranslationError(ValueError):
    pass


class _Names:
    """Hands out the SSA names `%0`, `%1`, ... that every block of the function shares."""

    def __init__(self) -> None:
        self.count = 0

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        self.count += 1
        return f"%{self.count - 1}"


class _Value(NamedTuple):
    ssa: str
    type: str  # "i32", "i1", "!dist.dist<i32>" or "!dist.dist<i1>"

    @property
    def dist(self) -> bool:
        return self.type.startswith("!dist.dist")


class _Slot(NamedTuple):
    ssa: str
    type: str  # what it holds, as a `_Value` type

    @property
    def dist(self) -> bool:
        return self.type.startswith("!dist.dist")

    @property
    def memref(self) -> str:
        return f"memref<{self.type}>"


class _Instance(NamedTuple):
    """One translation of a function, for the kinds of argument it was called with."""
    symbol: str
    result: str  # the result type in the IR
    kind: str  # "ref": returns a distribution it holds, as the slot; "fresh": a new range; "value": any other value
    writes: Tuple[int, ...]  # positions of the distribution parameters the function stores to


class _Module:
    """
    The functions of a program. A function has no declared types, so it is translated once for each combination of
    argument types it is called with: a definite argument is passed as a value, which makes the parameter a copy, and a
    distribution is passed as its slot, which makes the parameter the caller's own object.
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
        body = _Translator(self.width, _Names(), itertools.count(), {}, {}, self)
        arguments, parameter_slots = [], []
        for parameter, kind in zip(definition.params, kinds):
            argument = next(body.names)
            arguments.append(f"{argument}: {kind}")
            if kind.startswith("memref<"):  # a distribution: the caller's slot itself
                slot_id = next(body.ids)
                body.slots[slot_id] = _Slot(argument, kind[len("memref<"):-1])
                body.bindings[parameter.name] = slot_id
            else:  # a definite value: copied into a slot of the function's own
                slot_id = body.new_slot(parameter.name, kind)
                body.store(slot_id, _Value(argument, kind))
            parameter_slots.append(slot_id)
        *statements, final = definition.body
        if not isinstance(final, Return):
            raise TranslationError(f"{definition.name} must end with a return")
        for statement in statements:
            body.statement(statement)
        value, slot, result_kind = body.returned(final.value)
        returned = slot.ssa if slot else value.ssa
        result = slot.memref if slot else value.type
        writes = tuple(position for position, slot_id in enumerate(parameter_slots)
                       if kinds[position].startswith("memref<") and slot_id in body.written)
        lines = ["  " + line for line in [*body.lines, f"func.return {returned} : {result}"]]
        self.texts.append("\n".join([f"func.func private @{symbol}({', '.join(arguments)}) -> {result} {{", *lines, "}"]))
        self.active.pop()
        self.instances[key] = _Instance(symbol, result, result_kind, writes)
        return self.instances[key]


class _Translator:
    """
    Names are bound to slots by id. A definite value gets a slot of its own, so copying it copies the value. An alias binds
    a second name to a distribution's slot, so an update through either name changes the one object. The slots are shared by
    every region's translator, and each region has its own copy of the bindings, so a name first assigned in a branch or a
    loop body ends with it.
    """

    def __init__(self, width: int, names: _Names, ids: Iterator[int], slots: Dict[int, _Slot],
                 bindings: Dict[str, int], module: _Module) -> None:
        self.module = module
        self.width = width
        self.type = f"i{width}"
        self.names = names
        self.ids = ids
        self.lines: List[str] = []
        self.slots = slots
        self.bindings = bindings
        self.written: Set[int] = set()  # slots stored to in this region, including ones from outside it
        self.readonly: Set[int] = set()  # loop variables, shared with every region inside their loop
        self.depth = 0  # loops this translator is inside
        self.iterations = 1  # how often its statements run, from the loops it is inside
        self.nested = False  # inside the arm of a branch or the body of a loop, whose variables are its own copies
        self.region_start = 0  # slots with a lower id were made outside this region
        self.shadow: Optional[Dict[int, int]] = None  # in the arm of a conditional on a distribution: slot -> its copy
        self.outer: Set[int] = set()  # the slots that existed when that arm began

    def emit(self, text: str, type: str) -> _Value:
        """Add an operation that produces one value."""
        name = next(self.names)
        self.lines.append(f"{name} = {text}")
        return _Value(name, type)

    def distribution(self, element: str) -> str:
        return f"!dist.dist<{element}>"

    def constant(self, value: int) -> _Value:
        low, high = -(1 << (self.width - 1)), (1 << (self.width - 1)) - 1
        if not low <= value <= high:
            raise TranslationError(f"{value} does not fit in {self.type}")
        return self.emit(f"arith.constant {value} : {self.type}", self.type)

    def child(self) -> "_Translator":
        """A translator for the arm of a branch or the body of a loop, on a copy of the bindings."""
        inner = _Translator(self.width, self.names, self.ids, self.slots, dict(self.bindings), self.module)
        inner.depth, inner.iterations, inner.readonly = self.depth, self.iterations, self.readonly
        inner.nested, inner.region_start = True, len(self.slots)
        return inner

    def new_slot(self, name: Optional[str], type: str) -> int:
        slot_id = next(self.ids)
        ssa = next(self.names)
        self.lines.append(f"{ssa} = memref.alloca() : memref<{type}>")
        self.slots[slot_id] = _Slot(ssa, type)
        if name is not None:
            self.bindings[name] = slot_id
        return slot_id

    def load(self, slot_id: int) -> _Value:
        slot = self.slots[slot_id]
        return self.emit(f"memref.load {slot.ssa}[] : {slot.memref}", slot.type)

    def store(self, slot_id: int, value: _Value) -> None:
        slot = self.slots[slot_id]
        self.lines.append(f"memref.store {value.ssa}, {slot.ssa}[] : {slot.memref}")
        self.written.add(slot_id)

    def slot_id(self, name: str) -> int:
        """The slot a name reads and writes: in the arm of a conditional on a distribution, a copy of a definite variable."""
        if name not in self.bindings:
            raise TranslationError(f"{name} is used before it is assigned")
        slot_id = self.bindings[name]
        if self.shadow is None or slot_id not in self.outer or self.slots[slot_id].dist:
            return slot_id
        if slot_id not in self.shadow:
            copy = self.new_slot(None, self.slots[slot_id].type)
            self.store(copy, self.load(slot_id))
            self.written.discard(copy)  # filling the copy is not a change
            self.shadow[slot_id] = copy
        return self.shadow[slot_id]

    def read(self, name: str) -> _Value:
        return self.load(self.slot_id(name))

    def binary(self, node: BinOp) -> _Value:
        left, right = self.expression(node.left), self.expression(node.right)
        if left.dist or right.dist:
            name = {"+": "dist.addi", "-": "dist.subi", "<": "dist.slt"}.get(node.op)
            if name is None:
                raise TranslationError(f"operator {node.op!r} is not supported on a distribution yet")
            result = self.distribution("i1" if node.op == "<" else self.type)
            return self.emit(f"{name} {left.ssa}, {right.ssa} : {left.type}, {right.type} -> {result}", result)
        if node.op == "<":
            return self.emit(f"arith.cmpi slt, {left.ssa}, {right.ssa} : {self.type}", "i1")
        name = {"+": "arith.addi", "-": "arith.subi"}.get(node.op)
        if name is None:
            raise TranslationError(f"operator {node.op!r} is not supported yet")
        return self.emit(f"{name} {left.ssa}, {right.ssa} : {self.type}", self.type)

    def expression(self, node: Node) -> _Value:
        if isinstance(node, Number):
            if not isinstance(node.value, int):
                raise TranslationError("only integer constants are supported yet")
            return self.constant(node.value)
        if isinstance(node, Variable):
            return self.read(node.name)
        if isinstance(node, BinOp):
            return self.binary(node)
        if isinstance(node, Call):
            result, _ = self.call(node)
            if isinstance(result, _Slot):  # a distribution returned by reference is read where it is used
                return self.emit(f"memref.load {result.ssa}[] : {result.memref}", result.type)
            return result
        if isinstance(node, Distribution):
            start, stop = node.start.value, node.stop.value
            if stop <= start:
                raise TranslationError(f"the range {start}..{stop} is empty")
            if stop - start == 1:  # a range one value wide is an ordinary definite value
                return self.constant(start)
            return self.emit(f"dist.uniform {start}, {stop} : {self.distribution(self.type)}", self.distribution(self.type))
        raise TranslationError(f"{type(node).__name__} is not supported yet")

    def call(self, node: Call) -> Tuple[Union[_Value, _Slot], _Instance]:
        """
        A call is its body inlined with each parameter assigned from its argument. So a definite argument is copied, a
        plain distribution name is passed as its slot (the callee's updates are the caller's), and a computed distribution
        is made an independent instance first.
        """
        definition = self.module.functions.get(node.name)
        if definition is None:
            raise TranslationError(f"{node.name} is not a function")
        if len(node.args) != len(definition.params):
            raise TranslationError(f"{node.name} takes {len(definition.params)} argument(s) but is called with "
                                   f"{len(node.args)}")
        passed, kinds, slot_ids = [], [], []
        for argument in node.args:
            slot_id = None
            if isinstance(argument, Variable) and argument.name in self.bindings \
                    and self.slots[self.bindings[argument.name]].dist:
                slot_id = self.bindings[argument.name]
            else:
                value = self.expression(argument)
                if value.dist:
                    if not isinstance(argument, Distribution):
                        value = self.emit(f"dist.instance {value.ssa} : {value.type} -> {value.type}", value.type)
                    slot_id = self.new_slot(None, value.type)
                    self.store(slot_id, value)
                else:
                    passed.append(value)
                    kinds.append(value.type)
            if slot_id is not None:
                slot = self.slots[slot_id]
                passed.append(_Value(slot.ssa, slot.memref))
                kinds.append(slot.memref)
            slot_ids.append(slot_id)
        instance = self.module.instance(definition, kinds)
        for position in instance.writes:
            if self.shadow is not None and slot_ids[position] in self.outer:
                raise TranslationError(f"{node.name} updates a distribution, which a conditional on a distribution "
                                       "cannot do yet")
            self.written.add(slot_ids[position])
        text = (f"func.call @{instance.symbol}({', '.join(p.ssa for p in passed)}) : ({', '.join(kinds)}) -> "
                f"{instance.result}")
        result = self.emit(text, instance.result)
        if instance.kind == "ref":
            return _Slot(result.ssa, result.type[len("memref<"):-1]), instance
        return result, instance

    def returned(self, node: Node) -> Tuple[_Value, Optional[_Slot], str]:
        """What a function hands back: a distribution it holds (as its slot), or a value."""
        if isinstance(node, Variable) and node.name in self.bindings and self.slots[self.bindings[node.name]].dist:
            return _Value("", ""), self.slots[self.bindings[node.name]], "ref"
        if isinstance(node, Call):
            result, instance = self.call(node)
            if isinstance(result, _Slot):
                return _Value("", ""), result, "ref"
            return result, None, "fresh" if instance.kind == "fresh" else "value"
        value = self.expression(node)
        return value, None, "fresh" if isinstance(node, Distribution) else "value"

    def check_writable(self, name: str) -> None:
        if self.bindings[name] in self.readonly:
            raise TranslationError(f"the loop variable {name} cannot be assigned to inside its loop")

    def conditional(self, node: If) -> None:
        condition = self.expression(node.condition)
        if condition.dist:
            self.select(condition, node)
        else:
            self.branch_on_definite(condition, node)

    def branch_on_definite(self, condition: _Value, node: If) -> None:
        """An `scf.if` whose arms store to the variables they change."""
        arms = []
        for statements in (node.body, node.orelse):
            arm = self.child()
            for statement in statements:
                arm.statement(statement)
            arms.append(arm)
        self.absorb(arms, "a conditional on a definite value that changes a distribution is not supported yet")
        indented = [["  " + line for line in arm.lines] for arm in arms]
        self.lines += [f"scf.if {condition.ssa} {{", *indented[0], "} else {", *indented[1], "}"]

    def absorb(self, regions: Sequence["_Translator"], refusal: str) -> None:
        """Take over what the regions stored to outside themselves, refusing a distribution among it."""
        changed = {slot_id for region in regions for slot_id in region.written if slot_id < regions[0].region_start}
        if any(self.slots[slot_id].dist for slot_id in changed):
            raise TranslationError(refusal)
        self.written |= changed

    def select(self, condition: _Value, node: If) -> None:
        """
        Both arms are computed here, on copies of the definite variables they change, and one `dist.select` picks each
        changed variable's value, using the condition once. A changed variable then holds a distribution.
        """
        if self.shadow is not None or self.nested:
            raise TranslationError("a conditional on a distribution inside another conditional or a loop is not "
                                   "supported yet")
        arms = []
        for statements in (node.body, node.orelse):
            arm = _Translator(self.width, self.names, self.ids, self.slots, dict(self.bindings), self.module)
            arm.depth, arm.iterations, arm.readonly = self.depth, self.iterations, self.readonly
            arm.shadow, arm.outer = {}, set(self.slots)
            for statement in statements:
                arm.statement(statement)
            arms.append(arm)
            self.lines += arm.lines
        changed = sorted({slot_id for arm in arms for slot_id, copy in arm.shadow.items() if copy in arm.written})
        if not changed:
            return
        result_types = [self.distribution(self.slots[slot_id].type) for slot_id in changed]
        pick = [[self.load(arm.shadow.get(slot_id, slot_id)) for slot_id in changed] for arm in arms]
        results = next(self.names)
        count = f":{len(changed)}" if len(changed) > 1 else ""
        join = lambda values, key: ", ".join(key(value) for value in values)
        self.lines.append(
            f"{results}{count} = dist.select {condition.ssa}, ({join(pick[0], lambda v: v.ssa)}), "
            f"({join(pick[1], lambda v: v.ssa)}) : {condition.type}, ({join(pick[0], lambda v: v.type)}), "
            f"({join(pick[1], lambda v: v.type)}) -> ({', '.join(result_types)})")
        for index, slot_id in enumerate(changed):
            value = _Value(f"{results}#{index}" if len(changed) > 1 else results, result_types[index])
            for name in [n for n, bound in self.bindings.items() if bound == slot_id]:
                self.store(self.new_slot(name, value.type), value)

    def loop(self, node: ForLoop) -> None:
        """
        A loop with literal bounds is an `affine.for`, whose bounds must be static, so the trip count is classical. Its
        variable is a slot of its own that the body may only read. A pass unrolls the loop before the slots are promoted.
        A loop that runs no times is translated for its errors and then dropped.
        """
        start, stop = node.start.value, node.stop.value
        body = self.child()
        body.depth, body.iterations = self.depth + 1, self.iterations * max(stop - start, 1)
        if body.depth > MAX_DEPTH or body.iterations > MAX_ITERATIONS:
            raise TranslationError(f"loops nested {MAX_DEPTH} deep or running {MAX_ITERATIONS} iterations in all cannot "
                                   "be unrolled")
        index = next(self.names)
        counter = body.emit(f"arith.index_cast {index} : index to {self.type}", self.type)
        variable = body.new_slot(node.var, self.type)
        body.store(variable, counter)
        body.readonly = self.readonly | {variable}
        for statement in node.body:
            body.statement(statement)
        made = [body.slots[slot_id] for slot_id in range(body.region_start, len(body.slots))]
        if any(slot.dist for slot in made):
            raise TranslationError("a loop that changes or makes a distribution is not supported yet")
        self.absorb([body], "a loop that changes or makes a distribution is not supported yet")
        if stop > start:
            self.lines += [f"affine.for {index} = {start} to {stop} {{", *["  " + line for line in body.lines], "}"]

    def update(self, node: Assign) -> None:
        """`name += amount;` or `name -= amount;`, which the parser gives as `name = name + amount` marked in place."""
        slot_id = self.slot_id(node.name)
        self.check_writable(node.name)
        if self.shadow is not None and slot_id in self.outer and self.slots[slot_id].dist:
            raise TranslationError(f"{node.name} is a distribution, which a conditional on a distribution cannot update "
                                   "yet")
        target, amount = self.load(slot_id), self.expression(node.value.right)
        name = {"+": "add_assign", "-": "sub_assign"}[node.value.op]
        if not (target.dist or amount.dist):
            result = self.emit(f"arith.{'addi' if name == 'add_assign' else 'subi'} {target.ssa}, "
                               f"{amount.ssa} : {self.type}", self.type)
        else:
            if not target.dist:
                raise TranslationError(f"{node.name} holds an integer, which cannot be updated by a distribution yet")
            right = node.value.right
            if isinstance(right, Variable) and self.bindings.get(right.name) == self.bindings[node.name]:
                raise TranslationError(f"{node.name} {node.value.op}= ... adds a distribution to itself, which is not "
                                       "supported yet")
            result_type = self.distribution(self.type)
            result = self.emit(
                f"dist.{name} {target.ssa}, {amount.ssa} : {target.type}, {amount.type} -> {result_type}", result_type)
        self.store(slot_id, result)

    def assign(self, node: Assign) -> None:
        held = self.slots.get(self.bindings.get(node.name, -1))
        if held is not None and held.dist:
            raise TranslationError(f"assigning to {node.name} would drop the distribution it holds; only an in-place "
                                   "update changes one")
        if isinstance(node.value, Variable) and node.value.name in self.bindings \
                and self.slots[self.bindings[node.value.name]].dist:  # an alias of a distribution
            self.bindings[node.name] = self.bindings[node.value.name]
            return
        if isinstance(node.value, Call):
            result, instance = self.call(node.value)
            if isinstance(result, _Slot):  # a distribution the function returned by reference: an alias
                slot_id = next(self.ids)
                self.slots[slot_id] = result
                self.bindings[node.name] = slot_id
                return
            value, fresh = result, instance.kind == "fresh"
        else:
            value = self.expression(node.value)
            fresh = isinstance(node.value, Distribution)
        if value.dist and not fresh:  # a computed value: an independent instance
            result = value.type
            value = self.emit(f"dist.instance {value.ssa} : {result} -> {result}", result)
        if held is None:
            self.store(self.new_slot(node.name, value.type), value)
            return
        self.check_writable(node.name)
        if held.type != value.type:
            raise TranslationError(f"{node.name} holds {held.type}, which cannot take a value of type {value.type}")
        self.store(self.slot_id(node.name), value)

    def statement(self, node: Node) -> None:
        if self.shadow is not None and isinstance(node, (If, ForLoop)):
            raise TranslationError("a conditional or a loop inside a conditional on a distribution is not supported yet")
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
            returned = self.emit(f"dist.measure {returned.ssa} : {returned.type} -> {self.type}", self.type)
        body = ["  " + line for line in [*self.lines, f"func.return {returned.ssa} : {self.type}"]]
        return "\n".join([f"func.func @main() -> {self.type} {{", *body, "}"])


def translate(program: Sequence[Node], width: int = WIDTH) -> str:
    module = _Module(program, width)
    main = _Translator(width, _Names(), itertools.count(), {}, {}, module)
    text = main.program([node for node in program if not isinstance(node, FunctionDef)])
    return "\n\n".join([*module.texts, text])

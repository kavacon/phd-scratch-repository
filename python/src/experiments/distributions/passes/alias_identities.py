"""
Alias identities: the arm of an `if` that leaves a variable unchanged does nothing, and the arm that assigns it overwrites
the value the register holds.

Balancing gives an arm that does not assign `x` the identity `x = x`, and after renaming it is a copy of the old version
into the new one (`COPY x#2, x#1`), which would cost a register and a controlled copy of every bit. The new version is
instead the same register as the old one, so the identity is dropped, and in the arm that assigns it the old value is
cleared first, by the inverse of the steps that made it, under that arm's control. The cost is one controlled step per
set bit to clear what was there, where the copy was one per bit.

Made only when the old version is read by nothing after the `if`, was made from registers that are unchanged since,
was not made by a preparation (undoing one under a control needs it to be alone), and when the assigning arms do not read
it (writing into the register they read would need it to be zero). An arm of one condition is exclusive of another when
they test the same register for opposite values.
"""
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from compiler.compiler import ISA, compile_pass
from compiler.parser.instructions import Alloc, Ctrl, Dist, EndCtrl, ICtrl, IDist, Instruction, IOp, IsaProgram, Op, Reg, Update
from compiler.system import System
from experiments.distributions.passes.entanglement import reads
from experiments.distributions.passes.history import Controls, undo

WRITERS = (Op, Update, Dist, IOp, IDist)


def _base(name: str) -> str:
    return name.split("#")[0]


def _contexts(body: Sequence[Instruction]) -> Tuple[List[Controls], List[Tuple[int, ...]]]:
    """For each instruction, the controls it runs under and the indices of the instructions that opened them."""
    controls: List[Controls] = []
    openers: List[Tuple[int, ...]] = []
    stack: List[Tuple[object, bool, int]] = []
    for k, instruction in enumerate(body):
        if isinstance(instruction, EndCtrl):
            stack.pop()
        controls.append(tuple((c, h) for c, h, _ in stack))
        openers.append(tuple(o for _, _, o in stack))
        if isinstance(instruction, (Ctrl, ICtrl)):
            stack.append((instruction.control, isinstance(instruction, Ctrl), k))
    return controls, openers


def _exclusive(a: Controls, b: Controls) -> Optional[int]:
    """The depth at which two guards test the same register for opposite values, or None if they are not exclusive."""
    for i, ((c1, h1), (c2, h2)) in enumerate(zip(a, b, strict=False)):
        if c1 != c2:
            return None
        if h1 != h2:
            return i
    return None


def _rename(instruction: Instruction, old: str, new: str) -> Instruction:
    swap = lambda name: new if name == old else name  # noqa: E731
    operand = lambda o: Reg(swap(o.name)) if isinstance(o, Reg) else o  # noqa: E731
    if isinstance(instruction, (Op, IOp)):
        return replace(instruction, dst=swap(instruction.dst), sources=tuple(operand(s) for s in instruction.sources))
    if isinstance(instruction, Update):
        return replace(instruction, dst=swap(instruction.dst), source=operand(instruction.source))
    if isinstance(instruction, (Dist, IDist)):
        return replace(instruction, dst=swap(instruction.dst))
    if isinstance(instruction, (Ctrl, ICtrl)):
        return replace(instruction, control=operand(instruction.control))
    if isinstance(instruction, Alloc):
        return replace(instruction, register=swap(instruction.register))
    return instruction


def _alias_one(program: IsaProgram, body: List[Instruction]) -> Optional[IsaProgram]:
    controls, openers = _contexts(body)
    writes: Dict[str, List[int]] = {}
    readers: Dict[str, List[int]] = {}
    for k, instruction in enumerate(body):
        if isinstance(instruction, WRITERS):
            writes.setdefault(instruction.dst, []).append(k)
        for name in reads(instruction):
            readers.setdefault(name, []).append(k)
    for k, copy in enumerate(body):
        if not (isinstance(copy, Op) and copy.opcode == "copy" and len(copy.sources) == 1 and isinstance(copy.sources[0], Reg)
                and controls[k] and _base(copy.dst) == _base(copy.sources[0].name) and copy.dst != copy.sources[0].name):
            continue
        new, old = copy.dst, copy.sources[0].name
        identities = [j for j in writes.get(new, []) if body[j] == copy and controls[j]]
        others = [j for j in writes.get(new, []) if j not in identities]
        if not others or any(not controls[j] for j in others):
            continue
        distances = {j: [_exclusive(controls[i], controls[j]) for i in identities] for j in others}
        if any(d is None for ds in distances.values() for d in ds):
            continue  # an assigning arm that could run when an identity arm does
        first = min(identities + others)
        history = writes.get(old, [])
        if any(j >= first for j in history) or any(not isinstance(body[j], (Op, Update)) for j in history):
            continue
        if any(r >= first and r not in identities for r in readers.get(old, [])):
            continue  # the old version is still read after the conditional
        if any(old in reads(body[j]) for j in others):
            continue
        intact = all(
            not any(i < w < first for w in writes.get(source, []))
            for i in history for source in sorted(reads(body[i]) | {c.name for c, _ in controls[i] if isinstance(c, Reg)}))
        if not intact:
            continue
        # the arms that assign the new version clear the old value first, under their own controls: the outermost region
        # of each that excludes every identity arm
        arms = {openers[j][max(ds)] for j, ds in distances.items()}
        clear = undo([(body[j], controls[j]) for j in history])
        out: List[Instruction] = []
        for index, instruction in enumerate(body):
            if index in identities or (isinstance(instruction, Alloc) and instruction.register == new):
                continue
            out.append(_rename(instruction, new, old))
            if index in arms:
                out.extend(clear)
        result = Reg(old) if program.result == Reg(new) else program.result
        return IsaProgram(tuple(out), result)
    return None


@compile_pass(
    ISA, ISA,
    title="Alias identities",
    summary="Gives the arm of an if that leaves a variable unchanged no operation: the new version shares the old "
            "version's register, and the arm that assigns it clears the old value first.",
)
def alias_identities(program: IsaProgram, system: Optional[System] = None) -> IsaProgram:
    while True:
        body = [i for i in program.body if not (isinstance(i, Op) and i.opcode == "copy" and i.sources == (Reg(i.dst),))]
        program = replace(program, body=tuple(body))
        rewritten = _alias_one(program, body)
        if rewritten is None:
            return program
        program = rewritten

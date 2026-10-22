"""
Release values: free each register as soon as it can be, by undoing it, so that only what the result needs is held.

Release is unitary only (no measurement and reset). Each release is written as the inverse instructions and then `FREE`,
placed at the first point where all of these hold, trying the registers allocated last first:
- nothing later reads the register, and every register whose own history read it (its dependents) is already released or
  is kept for good, since undoing a dependent needs it intact;
- the register is not the result, and one of these ways of undoing it applies:
  - Reversal. The register is computed (its history has no `DIST`) and its sources are intact, so replaying its history
    backwards writes only that register. The entanglement graph does not matter: this cannot change any other register.
  - Inverse preparation. Its history starts with a `DIST` and nothing is entangled with it any more, so its updates
    can be undone and the preparation run backwards. Here the graph decides: a register entangled with something kept
    cannot be, because undoing it would change that something.
  - Inversion. The result was computed from it in one step that can be run backwards (adding or subtracting a constant,
    multiplying by an odd constant, a copy), so it is computed back from the result and zeroed. The result keeps the
    information the register held.
Registers none of this frees stay held to the end: the result, and whatever it is entangled with and cannot be inverted
from, which is the garbage a unitary-only program cannot avoid.

A register kept for good does not stop the registers it was computed from from being released, so the pass repeats with
the registers left held marked as kept until that stops changing anything. A register that is left held only because a
register computed from it is not released yet is not marked: it can be undone once that one is known to be kept, so the
registers marked are those that could not be undone at the end even with nothing waiting on them.

A controlled region is never split, so a release is only placed between regions. A step that ran under a control is
undone under the same control, since the register is zero wherever the condition did not hold. Every register a region
is controlled on counts as read by what the region writes, so it stays until those are released or kept. A register
that has been cleared by an inverse under a control is not released, since its history is no longer the steps that
wrote it. A program that already releases is returned unchanged with a warning.
"""
import warnings
from dataclasses import replace
from typing import Dict, List, Optional, Set, Tuple

from checking.simulate import normaliser
from compiler.compiler import ISA, compile_pass
from compiler.parser.instructions import (
    Alloc,
    Const,
    Ctrl,
    Dist,
    EndCtrl,
    Free,
    ICtrl,
    IDist,
    Instruction,
    IOp,
    IsaProgram,
    Op,
    Reg,
    Update,
    inverse,
)
from compiler.system import System
from experiments.distributions.passes.entanglement import State, read_after
from experiments.distributions.passes.history import Controls, undo

History = Dict[str, List[Tuple[int, Instruction]]]


def _sources(instruction: Instruction) -> List[str]:
    if isinstance(instruction, Op):
        return [s.name for s in instruction.sources if isinstance(s, Reg)]
    if isinstance(instruction, Update) and isinstance(instruction.source, Reg):
        return [instruction.source.name]
    return []


def _inverse_of_op(op: Op, register: str, result: str, width: int) -> Optional[Instruction]:
    """The instruction that computes `register` from `result`, if `result = op(register, ...)` can be run backwards."""
    constants = [s for s in op.sources if isinstance(s, Const)]
    variables = [s for s in op.sources if isinstance(s, Reg)]
    if variables != [Reg(register)]:
        return None
    here = Reg(result)
    if op.opcode == "copy":
        return IOp("copy", register, (here,))
    if len(constants) != 1 or isinstance(constants[0].value, bool) or not isinstance(constants[0].value, int):
        return None
    value, left = constants[0], op.sources[0] == Reg(register)
    if op.opcode == "add":
        return IOp("sub", register, (here, value))
    if op.opcode == "sub":
        return IOp("add", register, (here, value)) if left else IOp("sub", register, (value, here))
    if op.opcode == "mul" and value.value % 2:
        modulus = 1 << width
        odd = pow(value.value % modulus, -1, modulus)
        return IOp("mul", register, (here, Const(odd - modulus if odd >= modulus // 2 else odd)))
    return None


class _Releaser:
    def __init__(self, program: IsaProgram, width: int) -> None:
        self.program, self.body, self.width = program, program.body, width
        self.result = program.result.name if isinstance(program.result, Reg) else None
        self.after = read_after(program)
        self.history: History = {}
        self.controls: List[Controls] = []  # what each instruction runs under: a control and whether it had to hold
        self.cleared = {i.dst for i in self.body if isinstance(i, (IOp, IDist))}
        open_regions: List[Tuple[object, bool]] = []
        for k, instruction in enumerate(self.body):
            if isinstance(instruction, (Ctrl, ICtrl)):
                open_regions.append((instruction.control, isinstance(instruction, Ctrl)))
            elif isinstance(instruction, EndCtrl):
                open_regions.pop()
            self.controls.append(tuple(open_regions))
            if isinstance(instruction, (Dist, Op, Update)):
                self.history.setdefault(instruction.dst, []).append((k, instruction))
        self.dependents = {r: {d for d, h in self.history.items() if d != r and any(r in self.reads(k, i) for k, i in h)}
                           for r in (i.register for i in self.body if isinstance(i, Alloc))}

    def reads(self, k: int, instruction: Instruction) -> List[str]:
        """What an instruction in the history needs intact to be undone: its sources and the controls it runs under."""
        return _sources(instruction) + [c.name for c, _ in self.controls[k] if isinstance(c, Reg)]

    def run(self) -> List[Instruction]:
        kept: Set[str] = {self.result} if self.result else set()
        while True:
            out, released, waiting = self.simulate(kept)
            left = {r for r in self.dependents if r not in released and r not in waiting}
            if left <= kept:
                return out
            kept |= left

    def simulate(self, kept: Set[str]) -> Tuple[List[Instruction], Set[str], Set[str]]:
        state, out, released = State(), [], set()
        self.writes: Dict[str, List[int]] = {}
        for k, instruction in enumerate(self.body):
            out.append(instruction)
            state.apply(instruction)
            if isinstance(instruction, (Dist, Op, Update, IOp, IDist)):
                self.writes.setdefault(instruction.dst, []).append(k)
            point = k + 1
            while not state.controls:  # a region is never split, so release only between regions
                for register in reversed(state.held):
                    if register in kept or register in self.after[point]:
                        continue
                    if any(d not in released and d not in kept for d in self.dependents[register]):
                        continue
                    release = self.release(register, point, state)
                    if release is not None:
                        out.extend(release)
                        for instruction in release:
                            state.apply(instruction)
                        released.add(register)
                        break
                else:
                    break
        waiting = {r for r in state.held if r not in kept and r not in released
                   and self.release(r, len(self.body), state) is not None}  # only held up by what is computed from them
        return out, released, waiting

    def intact(self, register: str, point: int, state: State) -> bool:
        """Every register the history of `register` read still holds what it held then."""
        for k, instruction in self.history.get(register, []):
            for source in self.reads(k, instruction):
                if source not in state.held or any(k < j < point for j in self.writes.get(source, [])):
                    return False
        return True

    def release(self, register: str, point: int, state: State):
        history = self.history.get(register, [])
        if any(k >= point for k, _ in history):
            return None  # its own steps have not all run yet
        if register in self.cleared:
            return None
        if history and self.intact(register, point, state):
            prepared = any(isinstance(i, Dist) for _, i in history)
            if not prepared or state.group_of(register) == frozenset({register}):
                return undo([(i, self.controls[k]) for k, i in history]) + [Free(register)]
        computed_back = self.invert(register, point, state)
        return None if computed_back is None else [computed_back, Free(register)]

    def invert(self, register: str, point: int, state: State):
        """Compute the register back from the result, when the result was made from it in one step."""
        result = self.result
        if result is None or result == register or result not in state.held:
            return None
        made = self.history.get(result, [])
        if len(made) != 1 or not isinstance(made[0][1], Op):
            return None
        k, op = made[0]
        if k >= point or self.controls[k]:
            return None  # not computed yet, or only when a condition held, so it does not determine the register
        if any(k < j < point for j in self.writes.get(register, [])):
            return None  # the register changed after the result was computed from it
        return _inverse_of_op(op, register, result, self.width)


@compile_pass(
    ISA, ISA,
    title="Release values",
    summary="Frees each register as soon as it can be undone: by reversing its own steps, by running its preparation "
            "backwards once nothing is entangled with it, or by computing it back from the result.",
)
def release_values(program: IsaProgram, system: Optional[System] = None) -> IsaProgram:
    if any(isinstance(i, Free) for i in program.body):
        warnings.warn("release_values leaves a program that already frees registers as it is")
        return program
    width = (system or System()).register_width
    return replace(program, body=tuple(_Releaser(program, width).run()))

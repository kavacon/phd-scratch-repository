"""
Flatten nested controls: a region inside a region, controlled on two or more registers at once, becomes regions each
controlled on one register, using a register that holds the conjunction of the conditions.

    CTRL a                     $g = a AND b
      CTRL b                   CTRL $g
        X                        X
      ENDCTRL          ->      ENDCTRL
      Y                        CTRL a
    ENDCTRL                      Y
                               ENDCTRL

The regions of an `if` write the same registers and never both run, so the rewritten program writes the same registers
and the same values. A condition that is false is `NOT` of its register, which is computed once. The conjunction registers
are computed before the outermost region, and the release pass undoes them afterwards, since they are computed outside
any region. It trades the extra gates of an operation under several controls for the registers and gates of the
conjunctions, and it is a pass of its own so the two can be compared.

A nest is left alone if a register it is controlled on is not computed before it begins, or is written inside it.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from compiler.compiler import ISA, compile_pass
from compiler.parser.instructions import Alloc, Ctrl, Dist, EndCtrl, ICtrl, IDist, Instruction, IOp, IsaProgram, Op, Reg, Update
from compiler.system import System

Guard = Tuple[Tuple[str, bool], ...]  # (register, whether it must hold) for each condition, outermost first


@dataclass
class _Region:
    control: str
    polarity: bool
    items: List["_Region | Instruction"] = field(default_factory=list)


def _parse(body: List[Instruction], start: int) -> Tuple[Optional[_Region], int]:
    """The region opened at `start` and the index just past its end, or None if it holds something a region cannot."""
    opener = body[start]
    if not isinstance(opener.control, Reg):
        return None, start + 1
    region, k = _Region(opener.control.name, isinstance(opener, Ctrl)), start + 1
    while k < len(body) and not isinstance(body[k], EndCtrl):
        if isinstance(body[k], (Ctrl, ICtrl)):
            inner, k = _parse(body, k)
            if inner is None:
                return None, k
            region.items.append(inner)
        elif isinstance(body[k], (Op, Update, Dist, IOp, IDist)):
            region.items.append(body[k])
            k += 1
        else:
            return None, k + 1
    return region, k + 1


def _depth(region: _Region) -> int:
    return 1 + max((_depth(i) for i in region.items if isinstance(i, _Region)), default=0)


def _controls(region: _Region) -> List[str]:
    return [region.control] + [c for i in region.items if isinstance(i, _Region) for c in _controls(i)]


def _written(region: _Region) -> set:
    return {i.dst for i in region.items if not isinstance(i, _Region)} | {
        d for i in region.items if isinstance(i, _Region) for d in _written(i)}


class _Builder:
    def __init__(self, numbering: List[int]) -> None:
        self.numbering = numbering
        self.setup: List[Instruction] = []
        self.registers: Dict[object, str] = {}

    def fresh(self, prefix: str) -> str:
        self.numbering[0] += 1
        return f"${prefix}{self.numbering[0]}"

    def literal(self, control: str, polarity: bool) -> str:
        if polarity:
            return control
        if ("not", control) not in self.registers:
            name = self.registers[("not", control)] = self.fresh("n")
            self.setup += [Alloc(name), Op("not", name, (Reg(control),))]
        return self.registers[("not", control)]

    def guard_register(self, guard: Guard) -> str:
        """A register that holds when every condition in the guard does (the first condition's own register for one)."""
        if len(guard) == 1:
            return self.literal(*guard[0])
        if guard not in self.registers:
            before, last = self.guard_register(guard[:-1]), self.literal(*guard[-1])
            name = self.registers[guard] = self.fresh("g")
            self.setup += [Alloc(name), Op("and", name, (Reg(before), Reg(last)))]
        return self.registers[guard]

    def emit(self, region: _Region, guard: Guard) -> List[Instruction]:
        guard = guard + ((region.control, region.polarity),)
        out: List[Instruction] = []
        run: List[Instruction] = []

        def flush() -> None:
            if run:
                opener = Ctrl if len(guard) > 1 or guard[0][1] else ICtrl
                control = self.guard_register(guard) if len(guard) > 1 else guard[0][0]
                out.extend([opener(Reg(control)), *run, EndCtrl()])
                run.clear()

        for item in region.items:
            if isinstance(item, _Region):
                flush()
                out.extend(self.emit(item, guard))
            else:
                run.append(item)
        flush()
        return out


@compile_pass(
    ISA, ISA,
    title="Flatten nested controls",
    summary="Replaces a region controlled on several registers at once with regions controlled on one register that "
            "holds their conjunction, computed before the nest and undone after it.",
)
def flatten_controls(program: IsaProgram, system: Optional[System] = None) -> IsaProgram:
    body, out, numbering, k = list(program.body), [], [0], 0
    while k < len(body):
        if not isinstance(body[k], (Ctrl, ICtrl)):
            out.append(body[k])
            k += 1
            continue
        region, end = _parse(body, k)
        before = body[:k]
        ready = {i.register for i in before if isinstance(i, Alloc)} & {i.dst for i in before if isinstance(i, (Dist, Op, Update))}
        if (region is not None and _depth(region) > 1 and set(_controls(region)) <= ready
                and not set(_controls(region)) & _written(region)):
            builder = _Builder(numbering)
            flat = builder.emit(region, ())
            out.extend(builder.setup + flat)
        else:
            out.extend(body[k:end])
        k = end
    return IsaProgram(tuple(out), program.result)

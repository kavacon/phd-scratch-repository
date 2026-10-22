"""
Undoing the steps that wrote a register, under the controls they ran under.

A step written inside a region only ran where the region's condition held, and the register is zero elsewhere, so its
inverse runs under the same condition. `undo` takes the steps in the order they ran, each with the controls it ran under
(a register and whether it had to hold), and gives the inverse instructions in reverse, each wrapped in its own regions.
"""
from typing import List, Sequence, Tuple

from compiler.parser.instructions import Ctrl, EndCtrl, ICtrl, Instruction, Operand, inverse

Controls = Tuple[Tuple[Operand, bool], ...]


def undo(steps: Sequence[Tuple[Instruction, Controls]]) -> List[Instruction]:
    out: List[Instruction] = []
    for instruction, controls in reversed(steps):
        out += [Ctrl(control) if holds else ICtrl(control) for control, holds in controls]
        out.append(inverse(instruction))
        out += [EndCtrl()] * len(controls)
    return out

"""
The range analysis: for each integer value, the interval of values it can take, found by interval arithmetic from the
constants. This is the late narrowing pass of a classical compiler, kept as an analysis: it never changes the IR, and
lowering reads it to size each value's register. It is run on the IR as it stands, so it sees the effect of any
rewrite that came before (a folded constant has an exact interval).

A value the analysis does not understand gets the full range of its type, so the answer is always safe.
"""
from typing import Dict, NamedTuple

from xdsl.dialects.arith import AddiOp, ConstantOp, SubiOp
from xdsl.dialects.builtin import IntegerAttr, IntegerType, ModuleOp
from xdsl.ir import SSAValue


class Range(NamedTuple):
    low: int
    high: int

    @property
    def width(self) -> int:
        """The fewest bits of two's complement that hold every value in the range."""
        width = 1
        while not (-(1 << (width - 1)) <= self.low and self.high <= (1 << (width - 1)) - 1):
            width += 1
        return width


def full_range(value: SSAValue) -> Range:
    width = value.type.width.data
    return Range(-(1 << (width - 1)), (1 << (width - 1)) - 1)


def ranges(module: ModuleOp) -> Dict[SSAValue, Range]:
    found: Dict[SSAValue, Range] = {}

    def of(value: SSAValue) -> Range:
        return found.get(value) or full_range(value)

    for op in module.walk():
        if isinstance(op, ConstantOp) and isinstance(op.value, IntegerAttr) and isinstance(op.result.type, IntegerType):
            found[op.result] = Range(op.value.value.data, op.value.value.data)
        elif isinstance(op, AddiOp):
            left, right = of(op.lhs), of(op.rhs)
            found[op.result] = Range(left.low + right.low, left.high + right.high)
        elif isinstance(op, SubiOp):
            left, right = of(op.lhs), of(op.rhs)
            found[op.result] = Range(left.low - right.high, left.high - right.low)
        else:
            for result in op.results:
                if isinstance(result.type, IntegerType):
                    found[result] = full_range(result)

    # The IR type is an upper bound: a result that does not fit its type has wrapped there.
    return {value: interval if interval.width <= value.type.width.data else full_range(value)
            for value, interval in found.items()}

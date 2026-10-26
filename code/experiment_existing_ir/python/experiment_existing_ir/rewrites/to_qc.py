"""
Lower a translated program (`func` and `arith`) to MQT's QC dialect, one rewrite pattern per operation.

Every integer value becomes a register of qubits (a `memref` of `!qc.qubit`), as wide as the range analysis says the
value needs; the IR type is only an upper bound. Nothing is released until the end, where the result is measured into a
bit register and every register is deallocated.

- A constant is loaded with X gates on its set bits.
- A sum or difference is a fresh register: the first operand is copied in with CNOTs, the second is copied into a
  temporary register of the result's width, and the adder (or its inverse) adds the temporary into the result. Copies
  repeat the sign bit when the operand is narrower than the result and drop the high bits when it is wider, which is
  right modulo the result's width, and the result's width holds the result.
- The return measures the result's qubits, least significant first, into the function's result.

The patterns run in order, so by the time one matches an operation its operands have already become registers.
"""
from typing import Dict, List, Sequence, Tuple

from xdsl.context import Context
from xdsl.dialects import arith, func, memref
from xdsl.dialects.builtin import (FunctionType, IndexType, IntAttr, IntegerAttr, MemRefType, ModuleOp, StringAttr,
                                   UnitAttr, i1)
from xdsl.ir import Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern,
                                   op_type_rewrite_pattern)

from experiment_existing_ir.dialects import qc
from experiment_existing_ir.ranges import Range, ranges
from experiment_existing_ir.rewrites import adder


def allocate(width: int) -> memref.AllocOp:
    return memref.AllocOp([], [], MemRefType(qc.QubitType(), [width]))


def load(register: SSAValue) -> Tuple[List[Operation], List[SSAValue]]:
    """The operations that read each qubit out of a register, and the qubits."""
    ops: List[Operation] = []
    qubits: List[SSAValue] = []
    for k in range(register.type.shape.data[0].data):
        index = arith.ConstantOp(IntegerAttr(k, IndexType()))
        read = memref.LoadOp.get(register, [index.result])
        ops += [index, read]
        qubits.append(read.res)
    return ops, qubits


def copy(source: Sequence[SSAValue], target: Sequence[SSAValue]) -> List[Operation]:
    """The target takes the source: sign-extended if it is narrower, truncated if it is wider."""
    return [adder.cx(source[min(i, len(source) - 1)], qubit) for i, qubit in enumerate(target)]


class Constant(RewritePattern):
    def __init__(self, found: Dict[SSAValue, Range]):
        self.found = found

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ConstantOp, rewriter: PatternRewriter) -> None:
        if op.result not in self.found:  # the index constants this pass writes itself
            return
        register = allocate(self.found[op.result].width)
        reads, qubits = load(register.memref)
        value = op.value.value.data
        flips = [qc.XOp(operands=[q]) for k, q in enumerate(qubits) if (value >> k) & 1]  # a negative shift is two's complement
        rewriter.replace(op, [register, *reads, *flips], [register.memref])


class Sum(RewritePattern):
    """`arith.addi` and `arith.subi`."""

    def __init__(self, found: Dict[SSAValue, Range]):
        self.found = found

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.AddiOp | arith.SubiOp, rewriter: PatternRewriter) -> None:
        width = self.found[op.result].width
        result, operand, carry = allocate(width), allocate(width), allocate(1)
        ops: List[Operation] = [result, operand, carry]
        qubits: Dict[str, List[SSAValue]] = {}
        for name, register in (("left", op.lhs), ("right", op.rhs), ("result", result.memref),
                               ("operand", operand.memref), ("carry", carry.memref)):
            reads, qubits[name] = load(register)
            ops += reads
        ops += copy(qubits["left"], qubits["result"]) + copy(qubits["right"], qubits["operand"])
        arguments = qubits["operand"] + qubits["result"] + qubits["carry"]
        name = adder.adder_name(width)
        ops.append(adder.call(name, arguments) if isinstance(op, arith.AddiOp) else adder.call_inverse(name, arguments))
        module = op.get_toplevel_object()
        if not any(isinstance(sub, func.FuncOp) and sub.sym_name.data == name for sub in module.ops):
            module.body.block.add_op(adder.adder(width))
        rewriter.replace(op, ops, [result.memref])


class Return(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.ReturnOp, rewriter: PatternRewriter) -> None:
        main = op.parent_op()
        if main.sym_name.data != "main" or not (op.arguments and isinstance(op.arguments[0].type, MemRefType)):
            return
        registers = [alloc for alloc in main.walk() if isinstance(alloc, memref.AllocOp)]
        reads, qubits = load(op.arguments[0])
        width = len(qubits)
        bits = qc.BitsAllocOp(properties={"initialization": qc.BitInit(StringAttr("undefined"))},
                              result_types=[qc.BitRegisterType(IntAttr(width))])
        ops: List[Operation] = [*reads, bits]
        for k, qubit in enumerate(qubits):
            index = arith.ConstantOp(IntegerAttr(k, IndexType()))
            measure = qc.MeasureOp(operands=[qubit], result_types=[i1])
            ops += [index, measure, qc.BitsStoreOp(operands=[measure.result, bits.register, index.result])]
        ops += [memref.DeallocOp.get(alloc.memref) for alloc in registers]
        ops.append(func.ReturnOp(bits.register))
        main.function_type = FunctionType.from_lists([], [bits.register.type])
        main.attributes["mqt.entry_point"] = UnitAttr()
        rewriter.replace(op, ops)


class LowerToQC(ModulePass):
    name = "lower-to-qc"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        found = ranges(op)
        patterns = GreedyRewritePatternApplier([Constant(found), Sum(found), Return()])
        PatternRewriteWalker(patterns).rewrite_module(op)

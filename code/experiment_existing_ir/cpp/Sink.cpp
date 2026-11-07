// Compute each value late: move an operation to just before the first operation that uses its result.
//
// A register is allocated when the operation that produces it is lowered, so an operation that runs long before its result is used holds
// qubits for nothing in between. Moving it later keeps the program the same and shortens the live range of its result. It moves
// closer to its use only; the inputs it reads stay live until then, so the move is worth it when the result is wider than the inputs it
// keeps alive (a prepared distribution has no inputs at all). Which operations move: those that only compute a value, which is every
// operation of the dialect except the in-place updates, and a distribution made fresh (a range, a lift).
//
// It is not part of the default pipeline: over the golden programs it saves 3 qubits of 673 (the bound with no limit on moving is 8
// percent), because about half of what could move is blocked by the rule below, and release does slightly worse on some programs
// when the order changes. It is an optional stage (`--sink-to-use`).
//
// One rule limits it. An in-place update changes the value it takes, so an operation that reads that value has to run before the update:
// an operation does not move past the update that consumes one of its inputs.
#include "Dialects.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace existing_ir::dist;

namespace {
bool movable(Operation *op) {
  if (op->getNumResults() == 0 || op->getNumRegions() != 0 || op->hasTrait<OpTrait::IsTerminator>())
    return false;
  if (isa<AddAssignOp, SubAssignOp, arith::ConstantOp>(op))
    return false;
  if (isa<UniformOp, LiftOp>(op))  // fresh values: they allocate, but read and write nothing
    return true;
  return isMemoryEffectFree(op);
}

struct SinkToUse : PassWrapper<SinkToUse, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SinkToUse)
  StringRef getArgument() const final { return "sink-to-use"; }
  StringRef getDescription() const final { return "Move each computation to just before the first operation that uses its result"; }

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    if (!function.getBody().hasOneBlock())
      return;
    Block &block = function.getBody().front();
    SmallVector<Operation *> ops;
    for (Operation &op : block)
      ops.push_back(&op);
    // From the last operation to the first, so that an operation moves after the operations that use its result have moved.
    for (Operation *op : llvm::reverse(ops)) {
      if (!movable(op))
        continue;
      Operation *first = nullptr;  // the first user in the block
      for (Operation *user : op->getUsers())
        if (user->getBlock() == &block && (!first || user->isBeforeInBlock(first)))
          first = user;
      if (!first || std::next(op->getIterator()) == first->getIterator())
        continue;  // no user here, or already just before it
      // not past an update that consumes one of its inputs
      bool blocked = false;
      for (Operation *between = op->getNextNode(); between != first && !blocked; between = between->getNextNode())
        if (isa<AddAssignOp, SubAssignOp>(between))
          blocked = llvm::is_contained(op->getOperands(), between->getOperand(0));
      if (!blocked)
        op->moveBefore(first);
    }
  }
};
} // namespace

void existing_ir::registerSinkToUse() { PassRegistration<SinkToUse>(); }

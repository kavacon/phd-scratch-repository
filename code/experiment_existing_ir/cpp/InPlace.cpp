// Out-of-place sums and differences whose first operand dies there become in-place updates.
//
// `%c = dist.addi %t, %a` computes a new distribution from `%t` and `%a`. When `%t` has no other use, nothing can
// observe whether its register is kept or reused, so the sum can be done in `%t`'s own register: `dist.add_assign`. The
// result is correlated with the same inputs either way, so the meaning does not change; the circuit loses the result
// register and the copy into it. Addition is commutative, so the operands may be swapped when only the second one dies.
// A difference is converted only when its first operand dies.
//
// A value used twice by one operation (`a + a`) has two uses and is not converted. Dead operations count as uses, so run
// `canonicalize` first.
#include "Dialects.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace existing_ir::dist;

namespace {

struct InPlaceUpdates : PassWrapper<InPlaceUpdates, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InPlaceUpdates)
  StringRef getArgument() const final { return "in-place-updates"; }
  StringRef getDescription() const final {
    return "Turn a dist.addi or dist.subi whose first operand has no other use into an in-place update";
  }
  void getDependentDialects(DialectRegistry &registry) const override { registry.insert<DistDialect>(); }

  void runOnOperation() override {
    SmallVector<Operation *> candidates;
    getOperation().walk([&](Operation *op) {
      if (isa<AddIOp, SubIOp>(op))
        candidates.push_back(op);
    });
    auto dies = [](Value value) { return isa<DistType>(value.getType()) && value.hasOneUse(); };
    for (Operation *op : candidates) {
      Value lhs = op->getOperand(0), rhs = op->getOperand(1), target, amount;
      if (dies(lhs)) {
        target = lhs;
        amount = rhs;
      } else if (isa<AddIOp>(op) && dies(rhs)) {
        target = rhs;
        amount = lhs;
      } else {
        continue;
      }
      OpBuilder builder(op);
      Type type = op->getResult(0).getType();
      Operation *update = isa<AddIOp>(op) ? AddAssignOp::create(builder, op->getLoc(), type, target, amount).getOperation()
                                          : SubAssignOp::create(builder, op->getLoc(), type, target, amount).getOperation();
      op->getResult(0).replaceAllUsesWith(update->getResult(0));
      op->erase();
    }
  }
};

} // namespace

void existing_ir::registerInPlaceUpdates() { PassRegistration<InPlaceUpdates>(); }

// Resolving `dist.instance`, which marks a computed value assigned to a new variable.
//
// The decided meaning is an independent distribution with the law of the computed value, so `resolve-instances`
// replays the history of the value: every distribution operation it was computed from is cloned, with each
// `dist.uniform` giving a fresh distribution, and the instance becomes the clone. Plain values (constants and
// arithmetic on them) are definite and are shared, not cloned. The operations the original computation used are left
// without users, and `canonicalize` removes them.
//
// `tie-instances` is the alternative meaning, kept for comparison: the instance is its input, so the new variable stays
// tied to everything it was computed from.
#include "Dialects.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace existing_ir::dist;

namespace {

// The distribution operations a value was computed from, each after the operations it uses.
void collectHistory(Value value, SetVector<Operation *> &history) {
  Operation *definition = value.getDefiningOp();
  if (!definition || !isa<DistDialect>(definition->getDialect()) || history.contains(definition))
    return;
  for (Value operand : definition->getOperands())
    collectHistory(operand, history);
  history.insert(definition);
}

struct ResolveInstances : PassWrapper<ResolveInstances, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResolveInstances)
  StringRef getArgument() const final { return "resolve-instances"; }
  StringRef getDescription() const final {
    return "Make each dist.instance an independent distribution by replaying the history of its input";
  }
  void getDependentDialects(DialectRegistry &registry) const override { registry.insert<DistDialect>(); }

  void runOnOperation() override {
    SmallVector<InstanceOp> instances;
    getOperation().walk([&](InstanceOp instance) { instances.push_back(instance); });  // in program order
    for (InstanceOp instance : instances) {
      SetVector<Operation *> history;
      collectHistory(instance.getInput(), history);
      OpBuilder builder(instance);
      IRMapping mapping;
      for (Operation *operation : history)
        builder.clone(*operation, mapping);
      instance.getResult().replaceAllUsesWith(mapping.lookupOrDefault(instance.getInput()));
      instance.erase();
    }
  }
};

struct TieInstances : PassWrapper<TieInstances, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TieInstances)
  StringRef getArgument() const final { return "tie-instances"; }
  StringRef getDescription() const final { return "Make each dist.instance its input, keeping it tied to what it came from"; }
  void getDependentDialects(DialectRegistry &registry) const override { registry.insert<DistDialect>(); }

  void runOnOperation() override {
    SmallVector<InstanceOp> instances;
    getOperation().walk([&](InstanceOp instance) { instances.push_back(instance); });
    for (InstanceOp instance : instances) {
      instance.getResult().replaceAllUsesWith(instance.getInput());
      instance.erase();
    }
  }
};

} // namespace

void existing_ir::registerInstances() {
  PassRegistration<ResolveInstances>();
  PassRegistration<TieInstances>();
}

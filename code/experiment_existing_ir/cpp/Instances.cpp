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
    // What each replay cloned, kept so that the other arm of a conditional can use it.
    DenseMap<Value, std::pair<IRMapping, SetVector<Operation *>>> replays;
    for (InstanceOp instance : instances) {
      SetVector<Operation *> history;
      collectHistory(instance.getInput(), history);
      OpBuilder builder(instance);
      IRMapping mapping;
      // The two arms of a conditional are never both seen, so a value computed in one arm and in the other, from the same
      // operations, is computed once. The arms are told apart by the select that takes both: this instance's partner is the
      // value at the same place on the other side, and it must already be replayed (so that its clones come first).
      if (std::optional<Value> partner = otherArm(instance)) {
        auto found = replays.find(*partner);
        if (found != replays.end()) {
          SmallVector<Operation *> common;
          for (Operation *operation : history)
            if (found->second.second.contains(operation))
              common.push_back(operation);
          // Sharing only the distributions both arms start from saves no work and costs the arms their in-place updates, since
          // a shared value is used twice; it pays when an operation on them is shared too.
          bool worth = llvm::any_of(common, [](Operation *operation) { return !isa<UniformOp, LiftOp>(operation); });
          for (Operation *operation : worth ? ArrayRef<Operation *>(common) : ArrayRef<Operation *>())
            for (Value result : operation->getResults())
              mapping.map(result, found->second.first.lookup(result));
        }
      }
      for (Operation *operation : history)
        if (!mapping.contains(operation->getNumResults() ? operation->getResult(0) : Value()))
          builder.clone(*operation, mapping);
      Value replayed = mapping.lookupOrDefault(instance.getInput());
      instance.getResult().replaceAllUsesWith(replayed);
      replays[replayed] = {mapping, history};
      instance.erase();
    }
  }

  // The value the select puts at the same place on the other side from this instance, when this instance is an operand of one.
  static std::optional<Value> otherArm(InstanceOp instance) {
    for (OpOperand &use : instance.getResult().getUses())
      if (auto select = dyn_cast<SelectOp>(use.getOwner())) {
        unsigned index = use.getOperandNumber() - 1, n = select.getResults().size();  // operand 0 is the condition
        if (index < n)
          return select->getOperand(1 + n + index);
        return select->getOperand(1 + index - n);
      }
    return std::nullopt;
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

// The pattern rewrites on dist operations, as a pass so that each can be switched off.
//
// They are patterns of the dialect, applied by the greedy driver with the operations' folds, like `canonicalize`, but kept out of it so
// that a pipeline can omit them and compare the circuit with and without the source-level rewrites.
#include "Dialects.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace existing_ir::dist;

namespace {

struct DistRewrites : PassWrapper<DistRewrites, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DistRewrites)
  DistRewrites() = default;
  DistRewrites(const DistRewrites &other) : PassWrapper(other) {}
  Option<bool> combineSteps{*this, "combine-steps", llvm::cl::desc("(x + 1) + 2 is x + 3"), llvm::cl::init(true)};
  Option<bool> flatten{*this, "flatten", llvm::cl::desc("conditionals nested around the same value are one condition"), llvm::cl::init(true)};
  StringRef getArgument() const final { return "dist-rewrites"; }
  StringRef getDescription() const final { return "Source-level rewrites on dist operations: combine constant steps, flatten nested conditionals"; }
  void getDependentDialects(DialectRegistry &registry) const override { registry.insert<DistDialect, arith::ArithDialect>(); }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    existing_ir::populateDistRewrites(patterns, combineSteps, flatten);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace

void existing_ir::registerDistRewrites() { PassRegistration<DistRewrites>(); }

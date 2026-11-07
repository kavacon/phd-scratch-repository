// Comparisons that the ranges of their operands decide.
//
// `a < 10` of a value in 0..3 is true however the roll falls, so the comparison is a definite distribution, and the conditional
// on it is resolved by the stock folds. The ranges come from the integer range analysis through the dist operations' own
// `inferResultRanges`; only the comparisons are replaced, since every other operation with a single-valued result already
// has operands that are known and folds. A conditional on a comparison so decided then takes its holding arm.
#include "Dialects.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace existing_ir::dist;

namespace {

struct FoldByRange : PassWrapper<FoldByRange, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldByRange)
  StringRef getArgument() const final { return "fold-by-range"; }
  StringRef getDescription() const final { return "Replace a dist comparison that its operands' ranges decide by a constant"; }
  void getDependentDialects(DialectRegistry &registry) const override { registry.insert<DistDialect>(); }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<dataflow::IntegerRangeAnalysis>();
    if (failed(solver.initializeAndRun(module)))
      return signalPassFailure();
    SmallVector<std::pair<Operation *, APInt>> decided;
    module.walk([&](Operation *op) {
      if (!isa<SltOp, EqOp>(op))
        return;
      auto *lattice = solver.lookupState<dataflow::IntegerValueRangeLattice>(op->getResult(0));
      if (!lattice || lattice->getValue().isUninitialized())
        return;
      if (std::optional<APInt> value = lattice->getValue().getValue().getConstantValue())
        decided.push_back({op, *value});
    });
    for (auto &[op, value] : decided) {
      OpBuilder builder(op);
      Type element = cast<DistType>(op->getResult(0).getType()).getElement();
      auto point = PointAttr::get(op->getContext(), IntegerAttr::get(element, value));
      op->getResult(0).replaceAllUsesWith(ConstantOp::create(builder, op->getLoc(), op->getResult(0).getType(), point));
      op->erase();
    }
    // A conditional on a decided comparison takes the value of the arm that holds.
    SmallVector<SelectOp> selects;
    module.walk([&](SelectOp select) { selects.push_back(select); });
    for (SelectOp select : selects) {
      auto condition = select.getCondition().getDefiningOp<ConstantOp>();
      if (!condition)
        continue;
      bool holds = !condition.getValue().getValue().getValue().isZero();
      ValueRange taken = holds ? select.getTrueValues() : select.getFalseValues();
      select.replaceAllUsesWith(taken);
      select.erase();
    }
  }
};

}  // namespace

void existing_ir::registerFoldByRange() { PassRegistration<FoldByRange>(); }

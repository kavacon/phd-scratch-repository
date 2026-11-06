// Which register each value is held in, and how wide: the decisions the lowering used to make as it went, now an analysis of its
// own that writes its answer on the operations, so that later analyses (liveness, the peak of qubits live at once, release) and
// the lowering read the same plan.
//
// A value that is an integer or a distribution is held in one register of as many qubits as MLIR's own range analysis says its
// values need (a distribution's width is its element's: the analysis follows it through the interface on our operations). Two
// values share a register in two cases: a lift whose integer has no other use (the distribution is the integer's register), and
// an update whose target is not used after it (the result is the target's register, updated in place). A shared register is as
// wide as the widest of its values, so no update needs a wider one than its target has.
//
// The answer, as attributes on the operation that produces the values:
//   `existing_ir.register`: for each result, the number of its register (counted from 0 in order of first appearance), or -1 if
//                           the result is not held in a register;
//   `existing_ir.qubits`:   for each result, the qubits of its register, or 0;
//   `existing_ir.in_place`: on an update done in its target's register; `existing_ir.reuse`: on a lift that is its integer's register.
// (Reading a distribution, `dist.measure`, is its register too: the measurement is the program's return.)
#include "Dialects.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace existing_ir::dist;

void existing_ir::planRegisters(ModuleOp module) {
  MLIRContext *context = module.getContext();
  // A constant distribution is a lift of an integer constant again, which is what becomes a register.
  module.walk([&](ConstantOp constant) {
    OpBuilder builder(constant);
    Value integer = arith::ConstantOp::create(builder, constant.getLoc(), constant.getValue().getValue());
    constant.replaceAllUsesWith(LiftOp::create(builder, constant.getLoc(), constant.getType(), integer).getResult());
    constant.erase();
  });
  DataFlowSolver solver;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::SparseConstantPropagation>();
  solver.load<dataflow::IntegerRangeAnalysis>();
  if (failed(solver.initializeAndRun(module)))
    return;

  // The width each value needs, from the range analysis.
  DenseMap<Value, unsigned> widths;
  module.walk([&](Operation *op) {
    for (Value result : op->getResults()) {
      Type resultType = result.getType();
      if (auto dist = dyn_cast<DistType>(resultType))
        resultType = dist.getElement();
      auto type = dyn_cast<IntegerType>(resultType);
      auto *lattice = solver.lookupState<dataflow::IntegerValueRangeLattice>(result);
      if (!type || !lattice || lattice->getValue().isUninitialized())
        continue;
      const ConstantIntRanges &range = lattice->getValue().getValue();
      unsigned width = std::max(range.smin().getSignificantBits(), range.smax().getSignificantBits());
      widths[result] = std::min(width, type.getWidth());
    }
  });

  // The values that share a register (union-find), and the operations that make them share.
  DenseSet<Operation *> inPlace, reuse;
  DenseMap<Value, Value> parent;
  auto find = [&](Value value) {
    while (parent.contains(value) && parent[value] != value)
      value = parent[value];
    return value;
  };
  module.walk([&](Operation *op) {
    if (auto lift = dyn_cast<LiftOp>(op); lift && lift.getInput().hasOneUse()) {
      reuse.insert(op);
      parent.try_emplace(lift.getInput(), lift.getInput());
      parent.try_emplace(lift.getResult(), lift.getResult());
      parent[find(lift.getResult())] = find(lift.getInput());
    }
    if (auto measure = dyn_cast<MeasureOp>(op)) {  // reading a distribution is its register, measured at the return
      parent.try_emplace(measure.getInput(), measure.getInput());
      parent.try_emplace(measure.getResult(), measure.getResult());
      parent[find(measure.getResult())] = find(measure.getInput());
    }
    if (!isa<AddAssignOp, SubAssignOp>(op))
      return;
    bool laterUse = llvm::any_of(op->getOperand(0).getUsers(), [&](Operation *user) {
      return user != op && (user->getBlock() != op->getBlock() || !user->isBeforeInBlock(op));
    });
    if (laterUse)
      return;
    inPlace.insert(op);
    parent.try_emplace(op->getOperand(0), op->getOperand(0));
    parent.try_emplace(op->getResult(0), op->getResult(0));
    parent[find(op->getResult(0))] = find(op->getOperand(0));
  });
  DenseMap<Value, unsigned> registerWidths;
  for (auto &member : parent)
    registerWidths[find(member.first)] = std::max(registerWidths[find(member.first)], widths.lookup(member.first));

  // Write the plan, numbering the registers in order of first appearance.
  DenseMap<Value, int64_t> number;
  Builder builder(context);
  module.walk([&](Operation *op) {
    SmallVector<int64_t> registers, qubits;
    bool any = false;
    for (Value result : op->getResults()) {
      auto it = widths.find(result);
      if (it == widths.end()) {
        registers.push_back(-1);
        qubits.push_back(0);
        continue;
      }
      Value root = find(result);
      number.try_emplace(root, number.size());
      registers.push_back(number[root]);
      qubits.push_back(parent.contains(result) ? registerWidths[root] : it->second);
      any = true;
    }
    if (any) {
      op->setAttr("existing_ir.register", builder.getDenseI64ArrayAttr(registers));
      op->setAttr("existing_ir.qubits", builder.getDenseI64ArrayAttr(qubits));
    }
    if (inPlace.contains(op))
      op->setAttr("existing_ir.in_place", builder.getUnitAttr());
    if (reuse.contains(op))
      op->setAttr("existing_ir.reuse", builder.getUnitAttr());
  });
}

namespace {
struct PlanRegisters : PassWrapper<PlanRegisters, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanRegisters)
  StringRef getArgument() const final { return "plan-registers"; }
  StringRef getDescription() const final { return "Decide the register and the number of qubits of every integer and distribution value"; }
  void getDependentDialects(DialectRegistry &registry) const override { registry.insert<DistDialect, arith::ArithDialect>(); }
  void runOnOperation() override { existing_ir::planRegisters(getOperation()); }
};
} // namespace

void existing_ir::registerPlanRegisters() { PassRegistration<PlanRegisters>(); }

// Which distribution values are correlated with which, as groups.
//
// A forward dataflow analysis on MLIR's sparse framework. The fact held for each value is the set of `dist.uniform`
// operations it was computed from (its origins): a uniform has itself, any other operation's results have the union of
// its operands' origins, and a plain integer has none, so it is definite and joins nothing. The framework merges the
// sets wherever control flow joins and iterates to a fixed point, so the same analysis keeps working when conditionals
// and loops carry distributions.
//
// Two origins are in the same group when some value depends on both, taken transitively. The pass writes the answer on
// the operations it concerns, so tests and later passes can read it:
//   `dist.group`: the number of the operation's group, counted from 0 in order of first appearance;
//   on the function, `dist.groups` (how many) and `dist.largest_group` (the most values in one group).
// Groups only ever merge, so the answer is conservative. Whether a value is still needed later is a separate question
// (liveness) and is not asked here.
#include "Dialects.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace existing_ir::dist;

namespace {

// The lattice element: the uniforms a value was computed from. The bottom (a definite value) is the empty set.
struct Origins {
  std::set<Operation *> uniforms;

  static Origins join(const Origins &lhs, const Origins &rhs) {
    Origins result = lhs;
    result.uniforms.insert(rhs.uniforms.begin(), rhs.uniforms.end());
    return result;
  }
  bool operator==(const Origins &other) const { return uniforms == other.uniforms; }
  void print(raw_ostream &os) const { os << uniforms.size() << " origin(s)"; }
};
using OriginsLattice = dataflow::Lattice<Origins>;

class OriginsAnalysis : public dataflow::SparseForwardDataFlowAnalysis<OriginsLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult visitOperation(Operation *op, ArrayRef<const OriginsLattice *> operands,
                               ArrayRef<OriginsLattice *> results) override {
    Origins origins;
    if (isa<UniformOp>(op))
      origins.uniforms.insert(op);
    for (const OriginsLattice *operand : operands)
      origins = Origins::join(origins, operand->getValue());
    for (OriginsLattice *result : results)
      // A measured or otherwise plain result is definite again.
      propagateIfChanged(result, result->join(isa<DistType>(result->getAnchor().getType()) ? origins : Origins()));
    return success();
  }

  void setToEntryState(OriginsLattice *lattice) override { propagateIfChanged(lattice, lattice->join(Origins())); }
};

struct DistStatus : PassWrapper<DistStatus, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DistStatus)
  StringRef getArgument() const final { return "dist-status"; }
  StringRef getDescription() const final { return "Group the distribution values that are correlated"; }
  void getDependentDialects(DialectRegistry &registry) const override { registry.insert<DistDialect>(); }

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<OriginsAnalysis>();
    if (failed(solver.initializeAndRun(function)))
      return signalPassFailure();

    // Union-find over the uniforms: those some value depends on together are one group.
    DenseMap<Operation *, Operation *> parent;
    auto find = [&](Operation *uniform) {
      while (parent[uniform] != uniform)
        uniform = parent[uniform] = parent[parent[uniform]];
      return uniform;
    };
    auto origins = [&](Value value) -> const std::set<Operation *> & {
      static const std::set<Operation *> none;
      auto *lattice = solver.lookupState<OriginsLattice>(value);
      return lattice ? lattice->getValue().uniforms : none;
    };
    function.walk([&](UniformOp uniform) { parent[uniform] = uniform; });
    function.walk([&](Operation *op) {
      for (Value result : op->getResults())
        if (isa<DistType>(result.getType()))
          for (Operation *uniform : origins(result))
            parent[find(uniform)] = find(*origins(result).begin());
    });

    // Number the groups in order of first appearance and count the distribution values in each.
    DenseMap<Operation *, int64_t> number;
    DenseMap<Operation *, int64_t> sizes;
    Builder builder(&getContext());
    int64_t largest = 0;
    function.walk([&](Operation *op) {
      if (!isa<DistDialect>(op->getDialect()))
        return;
      Value carrier;  // a distribution value the operation produces or, failing that, reads
      for (Value result : op->getResults())
        if (isa<DistType>(result.getType()))
          carrier = result;
      for (Value operand : op->getOperands())
        if (!carrier && isa<DistType>(operand.getType()))
          carrier = operand;
      if (!carrier || origins(carrier).empty())
        return;
      Operation *root = find(*origins(carrier).begin());
      number.try_emplace(root, number.size());
      op->setAttr("dist.group", builder.getI64IntegerAttr(number[root]));
      for (Value result : op->getResults())
        if (isa<DistType>(result.getType()))
          largest = std::max(largest, ++sizes[root]);
    });
    function->setAttr("dist.groups", builder.getI64IntegerAttr(number.size()));
    function->setAttr("dist.largest_group", builder.getI64IntegerAttr(largest));
  }
};

} // namespace

void existing_ir::registerDistStatus() { PassRegistration<DistStatus>(); }

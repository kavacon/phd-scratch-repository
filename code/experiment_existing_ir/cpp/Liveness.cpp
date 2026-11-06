// How many qubits are live, from the plan of `plan-registers`.
//
// A register is live from the operation that first produces a value in it to the last operation that uses a value in it (MLIR's
// own `Liveness` finds where each value ends). Nothing here releases anything: it says what the lowering would need if every
// register were released the moment it was last used, next to what it needs while registers are held to the return.
//   on the function: `existing_ir.logical_qubits`: the qubits of all the registers, which is what holding every register costs;
//                    `existing_ir.peak_logical_qubits`: the most qubits live at once if each register is released at its last use;
//   on each operation: `existing_ir.live_qubits`: the qubits live while it runs (its operands and results included).
// The first two count the values themselves. The temporaries an operation uses while it runs (an operand copy, a carry) are the
// plan's `existing_ir.scratch`, and three ways of providing them are counted on the function:
//   `existing_ir.qubits_held`:      every register and every operation's temporaries held to the return (what the lowering does now);
//   `existing_ir.qubits_dedicated`: every register held, and one scratch region shared by all operations, as large as the
//                                   largest any operation needs (each operation leaves it clean);
//   `existing_ir.qubits_pool`:      registers released at their last use and temporaries freed when their operation ends, so the
//                                   most live at once is the worst operation's live registers plus its temporaries.
#include "Dialects.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

// The register intervals, the qubits live while each operation runs, and each operation's scratch, for a function that is one block.
bool existing_ir::computeTimeline(func::FuncOp function, QubitTimeline &timeline) {
  if (!function.getBody().hasOneBlock())
    return false;  // the program has been resolved to straight-line code; anything else is not measured
  Liveness liveness(function);
  DenseMap<Operation *, int64_t> index;
  function.walk([&](Operation *op) {
    index[op] = timeline.order.size();
    timeline.order.push_back(op);
  });
  for (Operation *op : timeline.order) {
    auto numbers = op->getAttrOfType<DenseI64ArrayAttr>("existing_ir.register");
    auto qubits = op->getAttrOfType<DenseI64ArrayAttr>("existing_ir.qubits");
    if (!numbers)
      continue;
    for (auto [result, number, n] : llvm::zip(op->getResults(), numbers.asArrayRef(), qubits.asArrayRef())) {
      if (number < 0)
        continue;
      Operation *end = liveness.getLiveness(op->getBlock())->getEndOperation(result, op);
      auto [it, fresh] = timeline.registers.try_emplace(number, existing_ir::QubitTimeline::Register{index[op], index[end], n});
      if (!fresh) {
        it->second.first = std::min(it->second.first, index[op]);
        it->second.last = std::max(it->second.last, index[end]);
      }
    }
  }
  for (Operation *op : timeline.order) {
    int64_t live = 0, here = index[op];
    for (auto &entry : timeline.registers)
      if (entry.second.first <= here && here <= entry.second.last)
        live += entry.second.qubits;
    auto scratch = op->getAttrOfType<IntegerAttr>("existing_ir.scratch");
    timeline.live.push_back(live);
    timeline.scratch.push_back(scratch ? scratch.getInt() : 0);
  }
  return true;
}

namespace {
struct QubitLiveness : PassWrapper<QubitLiveness, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QubitLiveness)
  StringRef getArgument() const final { return "qubit-liveness"; }
  StringRef getDescription() const final { return "Count the qubits live at each operation, from the register plan"; }

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    existing_ir::QubitTimeline timeline;
    if (!computeTimeline(function, timeline))
      return;
    Builder builder(&getContext());
    int64_t total = 0, scratchTotal = 0, scratchMax = 0, peak = 0, pool = 0;
    for (auto &entry : timeline.registers)
      total += entry.second.qubits;
    for (size_t i = 0; i < timeline.order.size(); ++i) {
      int64_t live = timeline.live[i], n = timeline.scratch[i];
      peak = std::max(peak, live);
      scratchTotal += n;
      scratchMax = std::max(scratchMax, n);
      pool = std::max(pool, live + n);
      if (live)
        timeline.order[i]->setAttr("existing_ir.live_qubits", builder.getI64IntegerAttr(live));
    }
    function->setAttr("existing_ir.qubits_held", builder.getI64IntegerAttr(total + scratchTotal));
    function->setAttr("existing_ir.qubits_dedicated", builder.getI64IntegerAttr(total + scratchMax));
    function->setAttr("existing_ir.qubits_pool", builder.getI64IntegerAttr(pool));
    function->setAttr("existing_ir.logical_qubits", builder.getI64IntegerAttr(total));
    function->setAttr("existing_ir.peak_logical_qubits", builder.getI64IntegerAttr(peak));
  }
};
} // namespace

void existing_ir::registerQubitLiveness() { PassRegistration<QubitLiveness>(); }

// How many qubits are live, from the plan of `plan-registers`.
//
// A register is live from the operation that first produces a value in it to the last operation that uses a value in it (MLIR's
// own `Liveness` finds where each value ends). Nothing here releases anything: it says what the lowering would need if every
// register were released the moment it was last used, next to what it needs while registers are held to the return.
//   on the function: `existing_ir.logical_qubits`: the qubits of all the registers, which is what holding every register costs;
//                    `existing_ir.peak_logical_qubits`: the most qubits live at once if each register is released at its last use;
//   on each operation: `existing_ir.live_qubits`: the qubits live while it runs (its operands and results included).
// These are the qubits of the values themselves. The temporaries an operation uses while it runs (an operand copy, a carry) are
// not values and are not counted here.
#include "Dialects.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct QubitLiveness : PassWrapper<QubitLiveness, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QubitLiveness)
  StringRef getArgument() const final { return "qubit-liveness"; }
  StringRef getDescription() const final { return "Count the qubits live at each operation, from the register plan"; }

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    if (!function.getBody().hasOneBlock())
      return;  // the program has been resolved to straight-line code; anything else is not measured
    Liveness liveness(function);
    SmallVector<Operation *> order;
    DenseMap<Operation *, int64_t> index;
    function.walk([&](Operation *op) {
      index[op] = order.size();
      order.push_back(op);
    });
    // The interval of operations in which each register is live, and its qubits.
    struct Interval { int64_t first, last, qubits; };
    DenseMap<int64_t, Interval> registers;
    for (Operation *op : order) {
      auto numbers = op->getAttrOfType<DenseI64ArrayAttr>("existing_ir.register");
      auto qubits = op->getAttrOfType<DenseI64ArrayAttr>("existing_ir.qubits");
      if (!numbers)
        continue;
      for (auto [result, number, n] : llvm::zip(op->getResults(), numbers.asArrayRef(), qubits.asArrayRef())) {
        if (number < 0)
          continue;
        Operation *end = liveness.getLiveness(op->getBlock())->getEndOperation(result, op);
        auto [it, fresh] = registers.try_emplace(number, Interval{index[op], index[end], n});
        if (!fresh) {
          it->second.first = std::min(it->second.first, index[op]);
          it->second.last = std::max(it->second.last, index[end]);
        }
      }
    }
    Builder builder(&getContext());
    int64_t total = 0, peak = 0;
    for (auto &entry : registers)
      total += entry.second.qubits;
    for (Operation *op : order) {
      int64_t live = 0, here = index[op];
      for (auto &entry : registers)
        if (entry.second.first <= here && here <= entry.second.last)
          live += entry.second.qubits;
      peak = std::max(peak, live);
      if (live)
        op->setAttr("existing_ir.live_qubits", builder.getI64IntegerAttr(live));
    }
    function->setAttr("existing_ir.logical_qubits", builder.getI64IntegerAttr(total));
    function->setAttr("existing_ir.peak_logical_qubits", builder.getI64IntegerAttr(peak));
  }
};
} // namespace

void existing_ir::registerQubitLiveness() { PassRegistration<QubitLiveness>(); }

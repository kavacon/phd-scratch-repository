// Put the operations of a resolved program in another order that means the same.
//
// What must stay in order is what the values say: an operation after the operations that produce its operands, and an operation that reads
// a value before the in-place update that consumes it. Everything else is free, and the order the lowering sees decides which registers are
// alive together, so the width, and which regions can share qubits. The policies:
//   source:     as written (the identity);
//   level:      all the operations that are ready, then all that become ready after them, and so on, which starts every region at once;
//   chain:      the operation that uses what was just made when there is one, which finishes a region before it starts the next;
//   random:     any ready operation, from a seed, to see how much the order matters;
//   pressure:   critical, but only among the operations that keep the qubits alive at once within a target, which is the system's
//               qubits when one is described and otherwise the most the order as written would have (each register counted from its
//               first operation to its last, an operation's temporaries while it runs, and a prepared register for good, since what
//               is computed from a die keeps it held); when none keeps within it, the one that adds least;
//   critical:   the ready operation with the longest chain of work still to follow it, as estimated from the widths of the plan
//               (`plan-registers` must have run), which is critical-path list scheduling. Gates on different qubits run side by side
//               whatever order they are written in, so what the order decides is who gets a free qubit first; the operations on the
//               longest chain get it, and the ones with slack are the ones that wait.
#include "Dialects.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <random>

using namespace mlir;
using namespace existing_ir::dist;

namespace {
struct ReorderOps : PassWrapper<ReorderOps, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReorderOps)
  ReorderOps() = default;
  ReorderOps(const ReorderOps &other) : PassWrapper(other) {}
  Option<std::string> policy{*this, "policy", llvm::cl::desc("source, level, chain, random, critical or pressure"), llvm::cl::init("source")};
  Option<int> seed{*this, "seed", llvm::cl::desc("seed of the random order"), llvm::cl::init(0)};
  StringRef getArgument() const final { return "reorder-ops"; }
  StringRef getDescription() const final { return "Reorder the independent operations of a resolved program"; }

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    if (!function.getBody().hasOneBlock() || policy == "source")
      return;
    Block &block = function.getBody().front();
    SmallVector<Operation *> ops;
    for (Operation &op : block)
      if (!op.hasTrait<OpTrait::IsTerminator>())
        ops.push_back(&op);
    DenseMap<Operation *, int> index;
    for (auto [i, op] : llvm::enumerate(ops))
      index[op] = i;
    // predecessors: the producers of the operands, and the readers of a value before the update that consumes it
    std::vector<std::set<int>> before(ops.size());
    for (auto [i, op] : llvm::enumerate(ops)) {
      for (Value operand : op->getOperands())
        if (Operation *producer = operand.getDefiningOp(); producer && index.count(producer))
          before[i].insert(index[producer]);
      if (isa<AddAssignOp, SubAssignOp>(op))
        for (Operation *user : op->getOperand(0).getUsers())
          if (user != op && index.count(user))
            before[i].insert(index[user]);
    }
    std::vector<int> remaining(ops.size());
    std::vector<std::vector<int>> after(ops.size());
    for (size_t i = 0; i < ops.size(); ++i) {
      remaining[i] = before[i].size();
      for (int b : before[i])
        after[b].push_back(i);
    }
    // how long each operation runs, roughly, and the longest chain of work from it to the end
    std::vector<double> priority(ops.size(), 0);
    if (policy == "critical" || policy == "pressure") {
      auto width = [&](Operation *op) -> double {
        if (auto q = op->getAttrOfType<DenseI64ArrayAttr>("existing_ir.qubits"); q && !q.empty() && q.asArrayRef()[0] > 0)
          return q.asArrayRef()[0];
        return 1;
      };
      auto duration = [&](Operation *op) -> double {
        double w = width(op);
        if (isa<AddIOp, SubIOp, AddAssignOp, SubAssignOp, arith::AddIOp, arith::SubIOp>(op))
          return 8 * w;
        if (isa<MulIOp, arith::MulIOp>(op))
          return 8 * w * w;
        if (isa<SltOp, arith::CmpIOp>(op))
          return 16 * w;
        if (isa<SelectOp, arith::SelectOp, EqOp>(op))
          return 4 * w;
        return w;
      };
      for (int i = ops.size() - 1; i >= 0; --i) {
        double below = 0;
        for (int a : after[i])  // (the order as written is a valid order, so the operations after this one have been done)
          below = std::max(below, priority[a]);
        priority[i] = duration(ops[i]) + below;
      }
    }
    // what each operation does to the qubits alive: the registers it first defines, those it touches (reads, or writes again), its temporaries
    DenseMap<Value, int64_t> registerOf;
    std::map<int64_t, int64_t> width;
    std::vector<std::vector<int64_t>> defines(ops.size()), touches(ops.size());
    std::vector<int64_t> scratch(ops.size(), 0);
    std::map<int64_t, int> touching;
    std::set<int64_t> permanent;  // prepared registers: a die that something kept is computed from stays held, so the model never frees one
    if (policy == "pressure") {
      for (size_t i = 0; i < ops.size(); ++i) {
        auto numbers = ops[i]->getAttrOfType<DenseI64ArrayAttr>("existing_ir.register");
        auto qubits = ops[i]->getAttrOfType<DenseI64ArrayAttr>("existing_ir.qubits");
        if (auto sc = ops[i]->getAttrOfType<IntegerAttr>("existing_ir.scratch"))
          scratch[i] = sc.getInt();
        if (numbers)
          for (auto [res, n, q] : llvm::zip(ops[i]->getResults(), numbers.asArrayRef(), qubits.asArrayRef()))
            if (n >= 0) {
              registerOf[res] = n;
              width[n] = q;
              if (isa<UniformOp>(ops[i]))
                permanent.insert(n);
            }
      }
      for (size_t i = 0; i < ops.size(); ++i) {
        std::set<int64_t> seen;
        for (Value operand : ops[i]->getOperands())
          if (registerOf.count(operand))
            seen.insert(registerOf[operand]);
        for (Value result : ops[i]->getResults())
          if (registerOf.count(result))
            seen.insert(registerOf[result]);
        touches[i].assign(seen.begin(), seen.end());
        for (int64_t r : seen)
          ++touching[r];
      }
    }
    // The qubits alive after the operations placed so far, as the order is built; and the most there would be for an order.
    std::set<int64_t> defined;
    int64_t live = 0;
    std::map<int64_t, int> left = touching;
    auto added = [&](int i) {
      int64_t n = 0;
      for (int64_t r : touches[i])
        if (!defined.count(r))
          n += width[r];
      return n;
    };
    auto place = [&](int i) {  // returns the qubits alive while it runs
      int64_t during = live + added(i) + scratch[i];
      for (int64_t r : touches[i]) {
        if (!defined.count(r)) {
          defined.insert(r);
          live += width[r];
        }
        if (--left[r] == 0 && !permanent.count(r))
          live -= width[r];
      }
      return during;
    };
    int64_t target = std::numeric_limits<int64_t>::max();
    if (policy == "pressure") {
      if (auto module = function->getParentOfType<ModuleOp>()) {
        FailureOr<Attribute> system = dlti::query(module, ArrayRef<StringRef>{"quantum", "qubits"});
        if (succeeded(system))
          if (auto attr = dyn_cast<IntegerAttr>(*system))
            target = attr.getInt();
      }
      if (target == std::numeric_limits<int64_t>::max()) {  // the most the order as written has
        int64_t most = 0;
        for (size_t i = 0; i < ops.size(); ++i)
          most = std::max(most, place(i));
        target = most;
        defined.clear();
        live = 0;
        left = touching;
      }
    }
    std::vector<int> ready, order;
    for (size_t i = 0; i < ops.size(); ++i)
      if (!remaining[i])
        ready.push_back(i);
    std::mt19937 random(seed);
    std::vector<int> next;  // for level: the ones that become ready
    int last = -1;
    while (!ready.empty()) {
      size_t pick = 0;
      if (policy == "random") {
        pick = random() % ready.size();
      } else if (policy == "pressure") {
        // by priority, among those that stay within the target; else the one that adds least
        int best = -1;
        int64_t cheapest = std::numeric_limits<int64_t>::max();
        for (size_t k = 0; k < ready.size(); ++k) {
          int64_t cost = live + added(ready[k]) + scratch[ready[k]];
          bool fits = cost <= target;
          int cb = best;
          if (fits) {
            if (cb < 0 || priority[ready[k]] > priority[ready[cb]] || (priority[ready[k]] == priority[ready[cb]] && ready[k] < ready[cb]))
              best = k;
          }
        }
        if (best < 0)
          for (size_t k = 0; k < ready.size(); ++k) {
            int64_t cost = live + added(ready[k]) + scratch[ready[k]];
            if (cost < cheapest || (cost == cheapest && ready[k] < ready[best])) {
              cheapest = cost;
              best = k;
            }
          }
        pick = best;
      } else if (policy == "critical") {
        for (size_t k = 1; k < ready.size(); ++k)
          if (priority[ready[k]] > priority[ready[pick]] || (priority[ready[k]] == priority[ready[pick]] && ready[k] < ready[pick]))
            pick = k;
      } else if (policy == "chain") {
        // the ready operation that comes after the one just made, if any; else the earliest
        pick = 0;
        bool found = false;
        for (size_t k = 0; k < ready.size(); ++k) {
          if (last >= 0 && before[ready[k]].count(last)) {
            pick = k;
            found = true;
            break;
          }
          if (!found && ready[k] < ready[pick])
            pick = k;
        }
      } else {  // level: ready operations in order, the newly ready ones wait for the next round
        pick = 0;
        for (size_t k = 1; k < ready.size(); ++k)
          if (ready[k] < ready[pick])
            pick = k;
      }
      int chosen = ready[pick];
      ready.erase(ready.begin() + pick);
      if (policy == "pressure")
        place(chosen);
      order.push_back(chosen);
      last = chosen;
      for (int a : after[chosen])
        if (--remaining[a] == 0)
          (policy == "level" ? next : ready).push_back(a);
      if (policy == "level" && ready.empty()) {
        ready = next;
        next.clear();
      }
    }
    Operation *terminator = block.getTerminator();
    for (int i : order)
      ops[i]->moveBefore(terminator);
  }
};
} // namespace

void existing_ir::registerReorderOps() { PassRegistration<ReorderOps>(); }

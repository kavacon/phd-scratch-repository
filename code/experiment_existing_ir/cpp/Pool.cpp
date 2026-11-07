// Physical qubits for the registers of a lowered program, from one pool.
//
// The lowering allocates a register of qubits for each value and each temporary and frees it when it is done with it, but the
// exporters do not reuse a freed register: each allocation becomes a qubit array of its own. This pass assigns the qubits itself.
// It walks `main` in order with a list of free qubits: an allocation takes the lowest free qubits (new ones when there are none),
// a deallocation gives its qubits back, and every use of a qubit of a register becomes a use of the qubit of the one pool register
// that it was given. A qubit is reused only after its register was freed, and the lowering frees a register only once it is back in
// the zero state, so a reused qubit starts as zero. Only the qubits some gate uses are given one.
//
// Which free qubit an allocation takes does not change how many qubits are needed but does change how much waits: an operation given
// the qubit another has just released has to wait for it, though it needs nothing from that operation. So an allocation takes a qubit the
// system has not used yet while there is one (`spread`; with no system described there is no limit to spread over), and otherwise the
// one that has been free longest.
//
// On `main`: `existing_ir.pool_qubits`, the size of the pool, and for each value register of the plan `existing_ir.qubits_of_<n>`,
// the pool qubits that hold it.
#include "Dialects.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Matchers.h"
#include <deque>
#include <set>

using namespace mlir;
using mlir::qc::QubitType;

namespace {
// The constant index of a load, if it has one.
std::optional<int64_t> constantIndex(memref::LoadOp load) {
  APInt value;
  if (matchPattern(load.getIndices().front(), m_ConstantInt(&value)))
    return value.getSExtValue();
  return std::nullopt;
}

bool isQubitRegister(Value reg) {
  auto type = dyn_cast<MemRefType>(reg.getType());
  return type && isa<QubitType>(type.getElementType());
}

struct AssignQubits : PassWrapper<AssignQubits, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AssignQubits)
  AssignQubits() = default;
  AssignQubits(const AssignQubits &other) : PassWrapper(other) {}
  Option<std::string> policy{*this, "policy", llvm::cl::desc("which qubit an allocation takes: aware (the free one that "
                             "is ready just in time, else a new one while the system has more, else the one ready soonest), spread "
                             "(a new one while the system has more, else the one free longest), oldest (the one free longest) or "
                             "lowest (the lowest free)"),
                             llvm::cl::init("aware")};
  StringRef getArgument() const final { return "assign-qubits"; }
  StringRef getDescription() const final { return "Give the registers of a lowered program qubits from one pool, reusing freed ones"; }
  void getDependentDialects(DialectRegistry &registry) const override { registry.insert<arith::ArithDialect, memref::MemRefDialect, mlir::qc::QCDialect>(); }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto main = module.lookupSymbol<func::FuncOp>("main");
    if (!main || !main.getBody().hasOneBlock())
      return;
    Block &block = main.getBody().front();

    // The qubit indices of each register that some operation loads.
    DenseMap<Value, std::set<int64_t>> used;
    block.walk([&](memref::LoadOp load) {
      if (auto index = constantIndex(load); index && isQubitRegister(load.getMemRef()))
        used[load.getMemRef()].insert(*index);
    });

    // Walk in order: give qubits at an allocation, take them back at a deallocation. The walk also runs the gates as early as their
    // qubits allow, one step for a gate and as many as the gates of the adder for a call, so that for each qubit it is known when it is
    // ready again. An allocation is then given a qubit that is ready by the time the first gate on the new register could start
    // for want of its other qubits: reusing it then costs no waiting. Which qubit is taken changes how much waits, never how many
    // qubits there are, unless the system has qubits to spare (see below).
    std::set<int64_t> lowestFree;    // the free qubits, for taking the lowest
    std::deque<int64_t> oldestFree;  // and in the order they were freed, for taking the one free longest
    std::vector<int64_t> freeQubits;  // and as a list, for taking the one that is ready when wanted
    bool lowest = policy == "lowest", aware = policy == "aware";
    int64_t next = 0;
    // "spread" and "aware": new qubits are taken while the system has them
    int64_t available = 0;
    bool spare = false;
    if (auto attr = main->getAttrOfType<IntegerAttr>("existing_ir.qubits_available"); attr && (policy == "spread" || aware)) {
      available = attr.getInt();
      spare = true;
    }
    DenseMap<int64_t, int64_t> ready;  // pool qubit -> when its last gate ends
    DenseMap<Value, DenseMap<int64_t, int64_t>> assigned;  // register -> its qubit index -> pool qubit
    auto qubitOf = [&](Value value) -> std::optional<int64_t> {
      auto load = value.getDefiningOp<memref::LoadOp>();
      if (!load || !isQubitRegister(load.getMemRef()))
        return std::nullopt;
      auto index = constantIndex(load);
      auto found = assigned.find(load.getMemRef());
      if (!index || found == assigned.end() || !found->second.contains(*index))
        return std::nullopt;
      return found->second[*index];
    };
    // the first gate on each register, whose other qubits say when the register is first wanted
    DenseMap<Value, Operation *> firstGate;
    for (Operation &op : block)
      if (isa<mlir::qc::QCDialect>(op.getDialect()))
        for (Value operand : op.getOperands())
          if (auto load = operand.getDefiningOp<memref::LoadOp>(); load && isQubitRegister(load.getMemRef()) && !firstGate.count(load.getMemRef()))
            firstGate[load.getMemRef()] = &op;
    auto cost = [&](Operation &op) -> int64_t {
      if (auto call = dyn_cast<mlir::qc::CallOp>(op))
        if (auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee()))
          return std::max<int64_t>(1, std::distance(callee.getBody().front().begin(), callee.getBody().front().end()) - 1);
      return 1;
    };
    auto take = [&](int64_t wanted) -> int64_t {
      if (aware) {
        int64_t best = -1;  // the free qubit that became ready latest but no later than wanted
        for (size_t i = 0; i < freeQubits.size(); ++i)
          if (ready[freeQubits[i]] <= wanted && (best < 0 || ready[freeQubits[i]] > ready[freeQubits[best]]))
            best = i;
        if (best < 0 && !(spare && next < available) && !freeQubits.empty()) {  // none in time and no spare: the one ready soonest
          best = 0;
          for (size_t i = 1; i < freeQubits.size(); ++i)
            if (ready[freeQubits[i]] < ready[freeQubits[best]])
              best = i;
        }
        if (best >= 0) {
          int64_t qubit = freeQubits[best];
          freeQubits.erase(freeQubits.begin() + best);
          return qubit;
        }
        return next++;
      }
      if (spare && next < available)
        return next++;
      if (lowest && !lowestFree.empty()) {
        int64_t qubit = *lowestFree.begin();
        lowestFree.erase(lowestFree.begin());
        return qubit;
      }
      if (!lowest && !oldestFree.empty()) {
        int64_t qubit = oldestFree.front();
        oldestFree.pop_front();
        return qubit;
      }
      return next++;
    };
    for (Operation &op : block) {
      if (auto alloc = dyn_cast<memref::AllocOp>(op); alloc && isQubitRegister(alloc)) {
        int64_t wanted = 0;
        if (auto gate = firstGate.find(alloc); aware && gate != firstGate.end())
          for (Value operand : gate->second->getOperands())
            if (auto other = qubitOf(operand))
              wanted = std::max(wanted, ready[*other]);
        for (int64_t index : used[alloc])
          assigned[alloc][index] = take(wanted);
      } else if (auto dealloc = dyn_cast<memref::DeallocOp>(op); dealloc && isQubitRegister(dealloc.getMemref())) {
        for (auto &[index, qubit] : assigned[dealloc.getMemref()]) {
          lowestFree.insert(qubit);
          oldestFree.push_back(qubit);
          freeQubits.push_back(qubit);
        }
      } else if (isa<mlir::qc::QCDialect>(op.getDialect())) {
        SmallVector<int64_t> qubits;
        for (Value operand : op.getOperands())
          if (auto qubit = qubitOf(operand))
            qubits.push_back(*qubit);
        int64_t start = 0;
        for (int64_t qubit : qubits)
          start = std::max(start, ready[qubit]);
        for (int64_t qubit : qubits)
          ready[qubit] = start + cost(op);
      }
    }

    // Rewrite: one pool register, and every load of a qubit of a register loads the qubit it was given.
    OpBuilder builder(&getContext());
    builder.setInsertionPointToStart(&block);
    Location loc = main.getLoc();
    Value pool = memref::AllocOp::create(builder, loc, MemRefType::get({next}, QubitType::get(&getContext())));
    SmallVector<memref::LoadOp> loads;
    block.walk([&](memref::LoadOp load) {
      if (isQubitRegister(load.getMemRef()))
        loads.push_back(load);
    });
    for (memref::LoadOp load : loads) {
      builder.setInsertionPoint(load);
      Value qubit = arith::ConstantIndexOp::create(builder, load.getLoc(), assigned[load.getMemRef()][*constantIndex(load)]);
      load.replaceAllUsesWith(memref::LoadOp::create(builder, load.getLoc(), pool, qubit).getResult());
      load.erase();
    }
    // Record where each value register is, then drop the registers.
    Builder attrs(&getContext());
    SmallVector<Operation *> registers;
    block.walk([&](memref::AllocOp alloc) {
      if (isQubitRegister(alloc) && alloc != pool.getDefiningOp())
        registers.push_back(alloc);
    });
    for (Operation *op : registers) {
      auto alloc = cast<memref::AllocOp>(op);
      auto number = alloc->getAttrOfType<IntegerAttr>("existing_ir.register");
      if (number) {
        SmallVector<int64_t> qubits;
        for (auto &[index, qubit] : assigned[alloc])
          qubits.push_back(qubit);
        llvm::sort(qubits);
        main->setAttr(("existing_ir.qubits_of_" + Twine(number.getInt())).str(), attrs.getDenseI64ArrayAttr(qubits));
      }
      for (Operation *user : llvm::make_early_inc_range(alloc->getUsers()))
        user->erase();  // its deallocation
      alloc->erase();
    }
    builder.setInsertionPoint(block.getTerminator());
    memref::DeallocOp::create(builder, loc, pool);
    main->setAttr("existing_ir.pool_qubits", attrs.getI64IntegerAttr(next));
    // The pool is what the program needs at once, so it is checked against the system's qubits here, after the releases.
    if (auto available = main->getAttrOfType<IntegerAttr>("existing_ir.qubits_available"); available && next > available.getInt()) {
      main.emitError() << "the program needs " << next << " qubits at once but the system has " << available.getInt() << " (over by "
                       << next - available.getInt() << ")";
      return signalPassFailure();
    }
  }
};
} // namespace

void existing_ir::registerAssignQubits() { PassRegistration<AssignQubits>(); }

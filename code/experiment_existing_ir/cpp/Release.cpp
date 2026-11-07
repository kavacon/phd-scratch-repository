// When each register may be released, and which are, from the plan and the liveness.
//
// To release a register is to return its qubits to zero so they can serve something else. A register is returned to zero by running
// the gates that computed it backwards. That is valid when its value is a classical function of registers that are still as they
// were: each input of each of its writing operations must still be held (not released) and must not have been changed since by an
// in-place update. A register that was born from a preparation (a range) is not a function of anything, and running its preparation
// backwards gives zero only if it is unentangled, so everything that depends on it, directly or through other registers, must have been
// released first; if one of them stays held, so does it. The register that holds the result is never released.
//
// Two strategies choose among the valid releases (the option `strategy`):
//   eager: each register is released at the earliest point it may be, after its last use. A register that others were computed
//          from waits until those that can be released have been, since it must be there to run them backwards;
//   lazy:  a register is released only when the qubits live at an operation would pass the system's qubits, and as few and as large
//          as will make it fit. With no system, nothing is released.
// The registers that stay held until the program ends are the garbage, and are reported, never an error: `existing_ir.garbage` on `main`
// lists, for each, its number, its qubits and why: 1 it is entangled with a held register that depends on it (a prepared register that
// something kept needs), 2 an input of it changes before it is last used, so it cannot be run backwards, 3 it could be released but
// the system did not need the qubits. (The register that holds the result is not garbage.)
// A register released has its number in `existing_ir.release_after` on the operation after which it is released, and every operation
// of `main` has its position in `existing_ir.id`. If the program does not fit however much is released, the pass says so.
#include "Dialects.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Pass/Pass.h"
#include <set>

using namespace mlir;

namespace {
constexpr int64_t NEVER = std::numeric_limits<int64_t>::max();

struct Writer {
  int64_t position;
  SmallVector<Value> inputs;  // operands held in other registers
  bool prepares;              // a preparation of a distribution, which is not a function of its inputs
};

struct ScheduleRelease : PassWrapper<ScheduleRelease, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleRelease)
  ScheduleRelease() = default;
  ScheduleRelease(const ScheduleRelease &other) : PassWrapper(other) {}
  Option<std::string> strategy{*this, "strategy", llvm::cl::desc("eager or lazy"), llvm::cl::init("lazy")};

  StringRef getArgument() const final { return "schedule-release"; }
  StringRef getDescription() const final { return "Decide which registers are released, and where, to fit the system's qubits"; }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto main = module.lookupSymbol<func::FuncOp>("main");
    existing_ir::QubitTimeline timeline;
    if (!main || !existing_ir::computeTimeline(main, timeline))
      return;
    Builder builder(&getContext());
    int64_t size = timeline.order.size();

    // The register of every value, and the operations that write each register, from the plan.
    DenseMap<Value, int64_t> registerOf, definedAt;
    std::map<int64_t, SmallVector<Writer>> writers;
    std::set<int64_t> registers;
    for (int64_t i = 0; i < size; ++i) {
      Operation *op = timeline.order[i];
      op->setAttr("existing_ir.id", builder.getI64IntegerAttr(i));
      auto numbers = op->getAttrOfType<DenseI64ArrayAttr>("existing_ir.register");
      if (!numbers)
        continue;
      for (auto [result, number] : llvm::zip(op->getResults(), numbers.asArrayRef()))
        if (number >= 0) {
          registerOf[result] = number;
          definedAt[result] = i;
          registers.insert(number);
        }
    }
    for (int64_t i = 0; i < size; ++i) {
      Operation *op = timeline.order[i];
      if (isa<existing_ir::dist::MeasureOp>(op))
        continue;  // reading a distribution is its register; it writes nothing
      auto numbers = op->getAttrOfType<DenseI64ArrayAttr>("existing_ir.register");
      if (!numbers)
        continue;
      for (auto [result, number] : llvm::zip(op->getResults(), numbers.asArrayRef())) {
        if (number < 0)
          continue;
        Writer writer{i, {}, isa<existing_ir::dist::UniformOp>(op)};
        for (Value operand : op->getOperands())
          if (registerOf.contains(operand) && registerOf[operand] != number)
            writer.inputs.push_back(operand);
        writers[number].push_back(writer);
      }
    }
    // The register the program returns is never released.
    DenseSet<int64_t> output;
    for (Operation *op : timeline.order)
      if (isa<func::ReturnOp>(op))
        for (Value operand : op->getOperands())
          if (registerOf.contains(operand))
            output.insert(registerOf[operand]);

    // The operation after which a value stops being what it was: the in-place update that consumes it.
    DenseMap<Value, int64_t> consumedAt;
    for (int64_t i = 0; i < size; ++i)
      if (timeline.order[i]->hasAttr("existing_ir.in_place"))
        consumedAt[timeline.order[i]->getOperand(0)] = i;
    auto lastUse = [&](int64_t r) { return timeline.registers[r].last; };

    // Who depends on whom: a register depends on the registers its writers read.
    std::map<int64_t, std::set<int64_t>> dependents;
    for (auto &[r, ws] : writers)
      for (Writer &writer : ws)
        for (Value input : writer.inputs)
          dependents[registerOf[input]].insert(r);
    auto prepared = [&](int64_t r) {
      return llvm::any_of(writers[r], [](Writer &w) { return w.prepares; });
    };
    // The latest point a register may be released: while every input is still as it was.
    auto latest = [&](int64_t r) {
      int64_t hi = size - 1;
      for (Writer &writer : writers[r])
        for (Value input : writer.inputs)
          if (consumedAt.contains(input))
            hi = std::min(hi, consumedAt[input] - 1);
      return hi;
    };
    // Everything that depends on a register, directly or through other registers. A prepared register stays entangled with a register
    // computed from it however many registers lie between, even when those between are themselves released.
    std::map<int64_t, std::set<int64_t>> reach;
    std::function<const std::set<int64_t> &(int64_t)> closure = [&](int64_t r) -> const std::set<int64_t> & {
      auto found = reach.find(r);
      if (found != reach.end())
        return found->second;
      std::set<int64_t> all;
      for (int64_t d : dependents[r]) {
        all.insert(d);
        const std::set<int64_t> &below = closure(d);
        all.insert(below.begin(), below.end());
      }
      return reach[r] = all;
    };
    // A register that can never be released: the output, one whose inputs change before it is last used, and a prepared one that
    // something that can never be released depends on, however indirectly.
    auto own = [&](int64_t r) { return output.contains(r) || latest(r) < lastUse(r); };
    std::map<int64_t, bool> never;
    auto impossible = [&](int64_t r) {
      auto found = never.find(r);
      if (found != never.end())
        return found->second;
      bool result = own(r);
      if (!result && prepared(r))
        for (int64_t d : closure(r))
          result |= own(d);
      return never[r] = result;
    };

    FailureOr<Attribute> system = dlti::query(module, ArrayRef<StringRef>{"quantum", "qubits"});
    auto available = succeeded(system) ? dyn_cast<IntegerAttr>(*system) : IntegerAttr();
    int64_t budget = available ? available.getInt() : NEVER;
    bool eager = strategy == "eager";

    DenseSet<int64_t> released;
    std::map<int64_t, int64_t> releasedAfter;
    SmallVector<int64_t> order;  // the registers in the order they are released, which has the registers computed from others first
    // Whether register r may be released right after operation t, given what has been released so far.
    std::function<bool(int64_t, int64_t)> valid = [&](int64_t r, int64_t t) {
      if (released.contains(r) || impossible(r) || t < lastUse(r) || t > latest(r))
        return false;
      for (Writer &writer : writers[r])
        for (Value input : writer.inputs)
          if (released.contains(registerOf[input]))
            return false;  // it has to be there to run this one backwards
      if (prepared(r))
        for (int64_t d : closure(r))
          if (!released.contains(d))
            return false;  // still entangled with something computed from it
      return true;
    };
    // Eagerly a register waits for the registers computed from it that can be released.
    auto waits = [&](int64_t r) {
      for (int64_t d : dependents[r])
        if (!released.contains(d) && !impossible(d))
          return true;
      return false;
    };
    auto release = [&](int64_t r, int64_t t) {
      released.insert(r);
      releasedAfter[r] = t;
      order.push_back(r);
    };
    auto heldDead = [&](int64_t i) {
      int64_t held = 0;
      for (int64_t r : registers)
        if (!released.contains(r) && timeline.registers[r].last < i)
          held += timeline.registers[r].qubits;
      return held;
    };

    int64_t needed = 0;
    for (int64_t i = 0; i < size; ++i) {
      if (eager && i > 0) {  // after operation i - 1
        bool changed = true;
        while (changed) {
          changed = false;
          for (auto it = registers.rbegin(); it != registers.rend(); ++it)
            if (valid(*it, i - 1) && !waits(*it)) {
              release(*it, i - 1);
              changed = true;
            }
        }
      }
      while (timeline.live[i] + timeline.scratch[i] + heldDead(i) > budget) {
        int64_t best = -1;  // the largest register that may be released now, after operation i - 1
        for (int64_t r : registers)
          if (i > 0 && valid(r, i - 1) && (best < 0 || timeline.registers[r].qubits > timeline.registers[best].qubits))
            best = r;
        if (best < 0) {
          main.emitError() << "the program needs " << timeline.live[i] + timeline.scratch[i] + heldDead(i) << " qubits at once at "
                           << timeline.order[i]->getName() << ", the first operation that does not fit, even releasing every register "
                           << "that may be released, but the system has " << budget;
          return signalPassFailure();
        }
        release(best, i - 1);
      }
      needed = std::max(needed, timeline.live[i] + timeline.scratch[i] + heldDead(i));
    }
    // A register released after the last operation is not released: the program has ended.
    DenseMap<int64_t, SmallVector<int64_t>> after;
    for (auto &[r, t] : releasedAfter)
      if (t < size - 1)
        after[t].push_back(r);
    for (auto &entry : after) {
      llvm::sort(entry.second);
      timeline.order[entry.first]->setAttr("existing_ir.release_after", builder.getDenseI64ArrayAttr(entry.second));
    }
    // What the emission needs once the operations are lowered and these are gone: (position, register) pairs in order, and the
    // positions of the operations that wrote each register released.
    SmallVector<int64_t> pairs;
    for (int64_t r : order)
      if (releasedAfter[r] < size - 1) {
        pairs.push_back(releasedAfter[r]);
        pairs.push_back(r);
        SmallVector<int64_t> positions;
        for (Writer &writer : writers[r])
          positions.push_back(writer.position);
        main->setAttr(("existing_ir.writers_" + Twine(r)).str(), builder.getDenseI64ArrayAttr(positions));
      }
    SmallVector<int64_t> garbage;
    int64_t blocked = 0, optional = 0;
    for (int64_t r : registers) {
      if (released.contains(r) || output.contains(r))
        continue;
      int64_t reason = latest(r) < lastUse(r) ? 2 : impossible(r) ? 1 : 3;
      garbage.insert(garbage.end(), {r, timeline.registers[r].qubits, reason});
      (reason == 3 ? optional : blocked) += timeline.registers[r].qubits;
    }
    main->setAttr("existing_ir.garbage", builder.getDenseI64ArrayAttr(garbage));
    main->setAttr("existing_ir.garbage_blocked_qubits", builder.getI64IntegerAttr(blocked));
    main->setAttr("existing_ir.garbage_optional_qubits", builder.getI64IntegerAttr(optional));
    main->setAttr("existing_ir.releases", builder.getDenseI64ArrayAttr(pairs));
    main->setAttr("existing_ir.released_registers", builder.getI64IntegerAttr(pairs.size() / 2));
    main->setAttr("existing_ir.qubits_needed", builder.getI64IntegerAttr(needed));
  }
};
} // namespace

void existing_ir::registerScheduleRelease() { PassRegistration<ScheduleRelease>(); }

// Which registers must be released, from the plan, the liveness and the system's qubits.
//
// A register is dead from the operation after its last use. Holding a dead register costs its qubits at every later operation; releasing
// it frees them. A release costs gates (the register has to be returned to zero), so the rule is to release only what must be
// released: a dead register is released when the qubits live at some later operation, with its temporaries and every dead register
// still held, would pass the system's qubits. When no system is described nothing needs releasing and every register stays held.
//
// The classes, as an array indexed by register number on `main` (`existing_ir.release_class`):
//   0 held:     left as it is to the return;
//   1 reuse:    released at its last use so that its qubits serve the operations after it, because the budget needs them.
// A register released "for fidelity" is one whose entanglement with a value still in use would change something computed from it
// later. Every operation here shuffles basis states and is applied after the preparation of the distributions, so nothing is ever
// computed that could interfere with it, and no register needs releasing for that reason; the class is added when the language has an
// operation that interferes (a Hadamard after the preparation).
//
// A register released has its number in `existing_ir.release_after` on the operation that is its last use. The registers to release
// are chosen largest first when several are dead, and `existing_ir.qubits_needed` on `main` is the most qubits live at once with
// those released. If even releasing every dead register leaves some operation over the budget, the program does not fit and the pass
// says by how much.
#include "Dialects.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct ClassifyRelease : PassWrapper<ClassifyRelease, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ClassifyRelease)
  StringRef getArgument() const final { return "classify-release"; }
  StringRef getDescription() const final { return "Say which registers must be released to fit the system's qubits"; }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto main = module.lookupSymbol<func::FuncOp>("main");
    existing_ir::QubitTimeline timeline;
    if (!main || !existing_ir::computeTimeline(main, timeline))
      return;
    FailureOr<Attribute> system = dlti::query(module, ArrayRef<StringRef>{"quantum", "qubits"});
    auto available = succeeded(system) ? dyn_cast<IntegerAttr>(*system) : IntegerAttr();
    int64_t budget = available ? available.getInt() : std::numeric_limits<int64_t>::max();

    DenseSet<int64_t> released;
    SmallVector<int64_t> numbers;
    for (auto &entry : timeline.registers)
      numbers.push_back(entry.first);
    // The qubits of the dead registers still held when operation `i` runs.
    auto heldDead = [&](size_t i) {
      int64_t held = 0;
      for (int64_t number : numbers)
        if (!released.contains(number) && timeline.registers[number].last < static_cast<int64_t>(i))
          held += timeline.registers[number].qubits;
      return held;
    };
    int64_t needed = 0;
    for (size_t i = 0; i < timeline.order.size(); ++i) {
      while (timeline.live[i] + timeline.scratch[i] + heldDead(i) > budget) {
        int64_t best = -1;  // the largest dead register still held
        for (int64_t number : numbers)
          if (!released.contains(number) && timeline.registers[number].last < static_cast<int64_t>(i) &&
              (best < 0 || timeline.registers[number].qubits > timeline.registers[best].qubits))
            best = number;
        if (best < 0) {
          main.emitError() << "the program needs " << timeline.live[i] + timeline.scratch[i] << " qubits at once at the first operation that does not fit, "
                           << timeline.order[i]->getName() << ", even with every dead register released, but the system has "
                           << budget;
          return signalPassFailure();
        }
        released.insert(best);
      }
      needed = std::max(needed, timeline.live[i] + timeline.scratch[i] + heldDead(i));
    }

    Builder builder(&getContext());
    SmallVector<int64_t> classes(numbers.empty() ? 0 : *llvm::max_element(numbers) + 1, 0);
    DenseMap<int64_t, SmallVector<int64_t>> after;  // by the position of the last use
    for (int64_t number : released) {
      classes[number] = 1;
      after[timeline.registers[number].last].push_back(number);
    }
    for (auto &entry : after) {
      llvm::sort(entry.second);
      timeline.order[entry.first]->setAttr("existing_ir.release_after", builder.getDenseI64ArrayAttr(entry.second));
    }
    main->setAttr("existing_ir.release_class", builder.getDenseI64ArrayAttr(classes));
    main->setAttr("existing_ir.qubits_needed", builder.getI64IntegerAttr(needed));
  }
};
} // namespace

void existing_ir::registerClassifyRelease() { PassRegistration<ClassifyRelease>(); }

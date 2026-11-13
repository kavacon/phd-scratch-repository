// The choice between the adder and the increment network for each constant added in place, made from what is known before any gate exists.
//
// The adder needs a copy of the constant and a carry, so w + 1 qubits more than the value; the network needs none, and costs more or less
// gates by the width and the bits of the constant (ConstantAdd.h). For each in-place addition or subtraction of an immediate constant:
//   - when the adder's scratch would take the qubits live at that operation over the machine's, the network (the only one that fits);
//   - otherwise the network if its predicted cost is no worse than the adder's on two-qubit gates and on depth, and the adder if not.
// The choice is written as `existing_ir.increment` on the operation (which the lowering follows) and its scratch is taken off when it is
// the network. `mode` can force either one.
#include "ConstantAdd.h"
#include "Dialects.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace existing_ir::dist;

namespace {
struct ChooseConstantAdd : PassWrapper<ChooseConstantAdd, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ChooseConstantAdd)
  ChooseConstantAdd() = default;
  ChooseConstantAdd(const ChooseConstantAdd &other) : PassWrapper(other) {}
  Option<std::string> mode{*this, "mode", llvm::cl::desc("auto, adder or network"), llvm::cl::init("auto")};

  StringRef getArgument() const final { return "constant-add"; }
  StringRef getDescription() const final { return "Choose, for each constant added in place, the adder or the increment network"; }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto main = module.lookupSymbol<func::FuncOp>("main");
    existing_ir::QubitTimeline timeline;
    if (!main || !existing_ir::computeTimeline(main, timeline))
      return;
    FailureOr<Attribute> system = dlti::query(module, ArrayRef<StringRef>{"quantum", "qubits"});
    auto available = succeeded(system) ? dyn_cast<IntegerAttr>(*system) : IntegerAttr();
    for (auto [i, op] : llvm::enumerate(timeline.order)) {
      if (!isa<AddAssignOp, SubAssignOp>(op) || !op->hasAttr("existing_ir.in_place"))
        continue;
      auto constant = op->getOperand(1).getDefiningOp<arith::ConstantOp>();
      if (!constant || !constant->hasAttr("existing_ir.immediate"))
        continue;
      unsigned width = op->getAttrOfType<DenseI64ArrayAttr>("existing_ir.qubits").asArrayRef()[0];
      int64_t value = cast<IntegerAttr>(constant.getValue()).getInt();
      existing_ir::Network network = existing_ir::bestNetwork(width, uint64_t(isa<SubAssignOp>(op) ? -value : value));
      bool useNetwork;
      if (mode == "network")
        useNetwork = true;
      else if (mode == "adder")
        useNetwork = false;
      else if (available && timeline.live[i] + timeline.scratch[i] > available.getInt())
        useNetwork = true;  // the adder does not fit
      else
        useNetwork = network.twoQubit <= existing_ir::adderTwoQubit(width) && network.depth <= existing_ir::adderDepth(width);
      if (useNetwork) {
        op->setAttr("existing_ir.increment", UnitAttr::get(&getContext()));
        op->removeAttr("existing_ir.scratch");
      } else {
        op->removeAttr("existing_ir.increment");
      }
    }
  }
};
} // namespace

void existing_ir::registerConstantAdd() { PassRegistration<ChooseConstantAdd>(); }

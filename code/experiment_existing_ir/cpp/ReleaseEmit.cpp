// Release the registers that `schedule-release` chose, on the lowered program.
//
// A register is returned to zero by running backwards the gates that computed it. The lowering tagged every operation it created
// with the position of the operation it replaced (`existing_ir.of`), so the gates that wrote a register are the ones tagged with the
// positions of its writers (`existing_ir.writers_<register>` on `main`). For each register to release, in the order scheduled and
// right after the operation scheduled, this pass runs each writer's gates backwards, the last writer first, as `qc.inv` of the same
// gates, which MQT's `unroll-modifiers` turns into the inverted gates in reverse order. A writer's temporaries were freed when it
// finished, so each is allocated again, in the zero state, for the gates run backwards and freed after them. Then the register is
// freed, and its qubits can be given to something else.
//
// With the option `uncompute=false` the gates are not emitted and the register is freed as it is. That is wrong, whatever it is
// reused for, and exists to show that a test depends on the uncomputation.
#include "Dialects.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using mlir::qc::QubitType;

namespace {
struct ReleaseRegisters : PassWrapper<ReleaseRegisters, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReleaseRegisters)
  ReleaseRegisters() = default;
  ReleaseRegisters(const ReleaseRegisters &other) : PassWrapper(other) {}
  Option<bool> uncompute{*this, "uncompute", llvm::cl::desc("run the gates that computed a register backwards"), llvm::cl::init(true)};

  StringRef getArgument() const final { return "release-registers"; }
  StringRef getDescription() const final { return "Return the registers chosen by schedule-release to zero and free them"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::qc::QCDialect, memref::MemRefDialect, arith::ArithDialect>();
  }

  static int64_t tag(Operation *op) {
    auto attr = op->getAttrOfType<IntegerAttr>("existing_ir.of");
    return attr ? attr.getInt() : -1;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto main = module.lookupSymbol<func::FuncOp>("main");
    if (!main || !main.getBody().hasOneBlock())
      return;
    auto pairs = main->getAttrOfType<DenseI64ArrayAttr>("existing_ir.releases");
    if (!pairs || pairs.empty())
      return;
    Block &block = main.getBody().front();
    OpBuilder b(&getContext());

    // The operations by the position they came from, and the allocation of each value register.
    DenseMap<int64_t, SmallVector<Operation *>> from;
    DenseMap<int64_t, memref::AllocOp> allocation;
    for (Operation &op : block) {
      if (int64_t t = tag(&op); t >= 0)
        from[t].push_back(&op);
      if (auto alloc = dyn_cast<memref::AllocOp>(op))
        if (auto number = alloc->getAttrOfType<IntegerAttr>("existing_ir.register"))
          allocation[number.getInt()] = alloc;
    }
    DenseMap<int64_t, Operation *> cursor;  // the last operation emitted after each position, so releases keep their order

    for (size_t k = 0; k + 1 < pairs.size(); k += 2) {
      int64_t position = pairs[k], number = pairs[k + 1];
      auto alloc = allocation.lookup(number);
      if (!alloc)
        continue;
      // right after the last operation of the program up to this position
      Operation *anchor = cursor.lookup(position);
      if (!anchor)
        for (Operation &op : block)
          if (int64_t t = tag(&op); t >= 0 && t <= position)
            anchor = &op;
      if (!anchor)
        continue;
      b.setInsertionPointAfter(anchor);
      Location loc = alloc.getLoc();
      auto writers = main->getAttrOfType<DenseI64ArrayAttr>(("existing_ir.writers_" + Twine(number)).str());
      if (uncompute && writers)
        for (int64_t writer : llvm::reverse(writers.asArrayRef()))
          runBackwards(b, loc, from[writer]);
      for (Operation *user : llvm::make_early_inc_range(alloc->getUsers()))
        if (isa<memref::DeallocOp>(user))
          user->erase();  // the one at the return
      cursor[position] = memref::DeallocOp::create(b, loc, alloc);
    }
  }

  // Run the gates among `ops` backwards, with the temporaries they used allocated again for the purpose.
  void runBackwards(OpBuilder &b, Location loc, ArrayRef<Operation *> ops) {
    SmallVector<Operation *> gates;
    SmallVector<memref::AllocOp> scratch;
    for (Operation *op : ops) {
      if (isa<mlir::qc::QCDialect>(op->getDialect()))
        gates.push_back(op);
      else if (auto alloc = dyn_cast<memref::AllocOp>(op); alloc && isa<QubitType>(alloc.getType().getElementType())) {
        auto role = alloc->getAttrOfType<StringAttr>("existing_ir.role");
        if (role && role.getValue() == "scratch")  // (the register this writer computed is not a temporary)
          scratch.push_back(alloc);
      }
    }
    if (gates.empty())
      return;
    // A new, zero register for each temporary of the writer, and the qubit loaded from it in place of the old one.
    DenseMap<Value, Value> fresh;  // old temporary register -> new
    SmallVector<Value> created;
    for (memref::AllocOp old : scratch) {
      Value reg = memref::AllocOp::create(b, loc, old.getType());
      reg.getDefiningOp()->setAttrs(old->getAttrDictionary());
      reg.getDefiningOp()->removeAttr("existing_ir.of");
      fresh[old] = reg;
      created.push_back(reg);
    }
    SmallVector<Value> outer, inner;  // the qubits of the gates, and what stands for each in the new body
    for (Operation *gate : gates)
      for (Value operand : gate->getOperands()) {
        if (!isa<QubitType>(operand.getType()) || llvm::is_contained(outer, operand))
          continue;
        outer.push_back(operand);
        auto load = operand.getDefiningOp<memref::LoadOp>();
        if (load && fresh.contains(load.getMemRef()))
          inner.push_back(memref::LoadOp::create(b, loc, fresh[load.getMemRef()], ValueRange(load.getIndices())));
        else
          inner.push_back(operand);
      }
    mlir::qc::InvOp::create(b, loc, inner, [&](ValueRange args) {
      IRMapping mapping;
      for (auto [from, to] : llvm::zip(outer, args))
        mapping.map(from, to);
      for (Operation *gate : gates)
        b.clone(*gate, mapping);
    });
    for (Value reg : created)
      memref::DeallocOp::create(b, loc, reg);
  }
};
} // namespace

void existing_ir::registerReleaseRegisters() { PassRegistration<ReleaseRegisters>(); }

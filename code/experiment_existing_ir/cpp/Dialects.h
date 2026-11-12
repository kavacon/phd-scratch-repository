#pragma once
// MQT's own QC and cbit dialects (built from source under code/build), plus our `dist` dialect.
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mqt/Dialect/CBit/IR/CBitOps.h"
#include "mqt/Dialect/QC/IR/QCOps.h"

#include "DistDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "DistTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "DistAttrs.h.inc"

#define GET_OP_CLASSES
#include "DistOps.h.inc"

namespace existing_ir {
// An in-place update (`add_assign`, `sub_assign`, and the `reflect` of an amplification) consumes its target: the result is the target updated. The target an operation
// consumes (null for any other operation), and whether it consumes a value, are said here once for every pass that needs them.
// (MLIR's destination-style interface says this for tensors and memrefs only; its verifier refuses any other type.)
inline mlir::Value consumedBy(mlir::Operation *op) {
  return llvm::isa<dist::AddAssignOp, dist::SubAssignOp, dist::ReflectOp>(op) ? op->getOperand(0) : mlir::Value();
}
inline bool consumes(mlir::Operation *op, mlir::Value value) { return consumedBy(op) == value; }
// When each register is live, from the plan (see Liveness.cpp). Indexes are positions in `order`.
struct QubitTimeline {
  struct Register { int64_t first, last, qubits; };
  llvm::SmallVector<mlir::Operation *> order;
  llvm::DenseMap<int64_t, Register> registers;  // by register number
  llvm::SmallVector<int64_t> live;              // qubits live while each operation runs, operands and results included
  llvm::SmallVector<int64_t> scratch;           // the temporaries each operation allocates
};
bool computeTimeline(mlir::func::FuncOp function, QubitTimeline &timeline);

// Whether an operation can take operand `index` as a constant written straight into its own qubits, so that a constant used only by such
// operations needs no register of its own: a sum or difference, a comparison, the amount of an update, the arms of a select, and the integer a
// distribution is lifted from (which becomes the starting state of the distribution's register).
inline bool acceptsImmediate(mlir::Operation *user, unsigned index) {
  using namespace dist;
  if (llvm::isa<mlir::arith::AddIOp, mlir::arith::SubIOp, mlir::arith::CmpIOp, AddIOp, SubIOp, SltOp, EqOp>(user))
    return true;
  if (llvm::isa<AddAssignOp, SubAssignOp>(user))
    return index == 1;
  if (llvm::isa<mlir::arith::SelectOp, SelectOp>(user))
    return index >= 1;
  return llvm::isa<LiftOp>(user);
}

// Registers the passes of this project.
void registerLowerToQC();
void registerDistStatus();
void registerInstances();
void registerInPlaceUpdates();
void registerFoldByRange();
void registerDistRewrites();
// The source-level rewrites that are patterns on dist operations: a chain of constant steps on one value is one step, and
// conditionals nested around the same value are one condition. Each is separate so that a pipeline can leave it out.
void populateDistRewrites(mlir::RewritePatternSet &patterns, bool combineSteps, bool flatten);
void registerResolveDist();
void registerPlanRegisters();
void registerQubitLiveness();
void registerScheduleRelease();
void registerReleaseRegisters();
void registerAssignQubits();
void registerSinkToUse();
void registerReorderOps();
// The plan of `plan-registers`, written on the operations of the module (see Registers.cpp).
void planRegisters(mlir::ModuleOp module);
} // namespace existing_ir

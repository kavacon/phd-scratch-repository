#pragma once
// MQT's own QC and cbit dialects (built from source under code/build), plus our `dist` dialect.
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
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
// When each register is live, from the plan (see Liveness.cpp). Indexes are positions in `order`.
struct QubitTimeline {
  struct Register { int64_t first, last, qubits; };
  llvm::SmallVector<mlir::Operation *> order;
  llvm::DenseMap<int64_t, Register> registers;  // by register number
  llvm::SmallVector<int64_t> live;              // qubits live while each operation runs, operands and results included
  llvm::SmallVector<int64_t> scratch;           // the temporaries each operation allocates
};
bool computeTimeline(mlir::func::FuncOp function, QubitTimeline &timeline);

// Registers the passes of this project.
void registerLowerToQC();
void registerDistStatus();
void registerInstances();
void registerInPlaceUpdates();
void registerResolveDist();
void registerPlanRegisters();
void registerQubitLiveness();
void registerClassifyRelease();
void registerAssignQubits();
// The plan of `plan-registers`, written on the operations of the module (see Registers.cpp).
void planRegisters(mlir::ModuleOp module);
} // namespace existing_ir

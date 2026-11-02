#pragma once
// MQT's own QC and cbit dialects (built from source under code/build), plus our `dist` dialect.
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
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

#define GET_OP_CLASSES
#include "DistOps.h.inc"

namespace existing_ir::dist {
// What a value of a distribution type is, from how it was made: a lift of an integer is definite, a range is a
// superposition, and an operation is one if an operand is. Until it is certain (an operation on lifts that has not folded
// yet, or a read that has not been forwarded) it is unknown.
enum class Kind { Definite, Superposed, Unknown };
Kind kindOf(mlir::Value value);
// Whether two values are the same: one value, or two lifted integers that are equal constants.
bool same(mlir::Value a, mlir::Value b);
// Whether a value is settled enough to compare: a superposition, or a lift of a constant.
bool settled(mlir::Value value);
} // namespace existing_ir::dist

namespace existing_ir {
// Registers the passes of this project.
void registerLowerToQC();
void registerDistStatus();
void registerInstances();
void registerInPlaceUpdates();
void registerCheckResolved();
} // namespace existing_ir

#pragma once
// MQT's own QC and cbit dialects (built from source in external/), plus our one `dist` dialect.
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mqt/Dialect/CBit/IR/CBitOps.h"
#include "mqt/Dialect/QC/IR/QCOps.h"

#include "DistDialect.h.inc"
#define GET_OP_CLASSES
#include "DistOps.h.inc"

namespace existing_ir {
// Registers the passes of this project.
void registerLowerToQC();
} // namespace existing_ir

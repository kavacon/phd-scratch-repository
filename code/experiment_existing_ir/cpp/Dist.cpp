#include "Dialects.h"

using namespace mlir;

#include "DistDialect.cpp.inc"
#define GET_OP_CLASSES
#include "DistOps.cpp.inc"

void existing_ir::dist::DistDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "DistOps.cpp.inc"
      >();
}

// The values are start .. stop - 1, which is what lets the range analysis size the register.
void existing_ir::dist::UniformOp::inferResultRanges(ArrayRef<ConstantIntRanges>, SetIntRangeFn setResultRange) {
  unsigned width = getType().getIntOrFloatBitWidth();
  setResultRange(getResult(), ConstantIntRanges::fromSigned(APInt(width, getStart(), true), APInt(width, getStop() - 1, true)));
}

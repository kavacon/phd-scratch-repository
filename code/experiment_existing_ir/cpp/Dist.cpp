#include "Dialects.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace existing_ir::dist;

#include "DistDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "DistTypes.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "DistAttrs.cpp.inc"
#define GET_OP_CLASSES
#include "DistOps.cpp.inc"

namespace {
// Every operation may be inlined: none of them depends on where it sits, so a call that holds them can be replaced by the
// callee's body.
struct DistInliner : DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final { return true; }
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final { return true; }
};
} // namespace

void DistDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "DistTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "DistAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "DistOps.cpp.inc"
      >();
  addInterfaces<DistInliner>();
}

// The integer a type stands for: the element of a distribution, or the integer itself.
static Type elementOf(Type type) {
  if (auto dist = dyn_cast<DistType>(type))
    return dist.getElement();
  return type;
}

LogicalResult UniformOp::verify() {
  if (int32_t(getStop()) <= int32_t(getStart()))
    return emitOpError("needs stop greater than start");
  if (!elementOf(getResult().getType()).isSignlessInteger())
    return emitOpError("needs an integer element type");
  return success();
}

// A binary operation has one element type throughout, and arithmetic on plain integers stays in `arith`.
template <typename Op> static LogicalResult verifyBinary(Op op, bool comparison) {
  if (!isa<DistType>(op.getLhs().getType()) && !isa<DistType>(op.getRhs().getType()))
    return op.emitOpError("needs a distribution operand; use arith on plain integers");
  if (elementOf(op.getLhs().getType()) != elementOf(op.getRhs().getType()))
    return op.emitOpError("needs operands with the same element type");
  Type expected = comparison ? Type(IntegerType::get(op.getContext(), 1)) : elementOf(op.getLhs().getType());
  if (elementOf(op.getResult().getType()) != expected)
    return op.emitOpError("has the wrong result element type: expected ") << expected;
  return success();
}
LogicalResult AddIOp::verify() { return verifyBinary(*this, false); }
LogicalResult SubIOp::verify() { return verifyBinary(*this, false); }
LogicalResult MulIOp::verify() { return verifyBinary(*this, false); }
LogicalResult SltOp::verify() { return verifyBinary(*this, true); }
LogicalResult EqOp::verify() { return verifyBinary(*this, true); }
LogicalResult AndIOp::verify() { return verifyBinary(*this, false); }
LogicalResult OrIOp::verify() { return verifyBinary(*this, false); }
LogicalResult XOrIOp::verify() { return verifyBinary(*this, false); }

LogicalResult LiftOp::verify() {
  if (elementOf(getResult().getType()) != getInput().getType())
    return emitOpError("needs a distribution whose element type is the integer's type");
  return success();
}

LogicalResult InstanceOp::verify() {
  if (getInput().getType() != getResult().getType())
    return emitOpError("needs a result of the type of its input");
  return success();
}

// An update consumes its target: it is changed in place, so every other use of it must come before the update, in the
// same block, and the amount must be a different value.
template <typename Op> static LogicalResult verifyUpdate(Op op) {
  if (!isa<DistType>(op.getTarget().getType()) && !isa<DistType>(op.getAmount().getType()))
    return op.emitOpError("needs a distribution operand; use arith on plain integers");
  if (elementOf(op.getTarget().getType()) != elementOf(op.getAmount().getType()) ||
      elementOf(op.getResult().getType()) != elementOf(op.getTarget().getType()))
    return op.emitOpError("needs operands and a result with the same element type");
  if (op.getTarget() == op.getAmount())
    return op.emitOpError("cannot add a value to itself in place");
  // Only a distribution is consumed. A use that the update dominates would see the updated register, so it is refused; a
  // use before it, or in the other arm of a branch, does not.
  if (isa<DistType>(op.getTarget().getType())) {
    DominanceInfo dominance(op->getParentOp());
    for (OpOperand &use : op.getTarget().getUses())
      if (use.getOwner() != op && dominance.properlyDominates(op.getOperation(), use.getOwner()))
        return use.getOwner()->emitOpError("uses a distribution after it was updated in place");
  }
  return success();
}
LogicalResult AddAssignOp::verify() { return verifyUpdate(*this); }
LogicalResult SubAssignOp::verify() { return verifyUpdate(*this); }

LogicalResult SelectOp::verify() {
  if (!elementOf(getCondition().getType()).isInteger(1))
    return emitOpError("needs a condition that is a distribution of one-bit integers");
  if (getTrueValues().size() != getFalseValues().size() || getTrueValues().size() != getResults().size())
    return emitOpError("needs as many true values, false values and results");
  for (auto [t, f, r] : llvm::zip(getTrueValues(), getFalseValues(), getResults()))
    if (elementOf(t.getType()) != elementOf(f.getType()) || elementOf(t.getType()) != cast<DistType>(r.getType()).getElement())
      return emitOpError("needs each pair of values and its result to have the same element type");
  return success();
}

LogicalResult MeasureOp::verify() {
  if (elementOf(getInput().getType()) != getResult().getType())
    return emitOpError("needs a result of the distribution's element type");
  return success();
}

// The ranges of the results, from the ranges of the operands that the analysis passes in. A plain integer operand and a
// distribution operand carry their ranges the same way.
static unsigned widthOf(Type type) { return elementOf(type).getIntOrFloatBitWidth(); }

void UniformOp::inferResultRanges(ArrayRef<ConstantIntRanges>, SetIntRangeFn setResultRange) {
  unsigned width = widthOf(getResult().getType());
  setResultRange(getResult(), ConstantIntRanges::fromSigned(APInt(width, int32_t(getStart()), true), APInt(width, int32_t(getStop()) - 1, true)));
}
void AddIOp::inferResultRanges(ArrayRef<ConstantIntRanges> ranges, SetIntRangeFn setResultRange) {
  setResultRange(getResult(), intrange::inferAdd(ranges));
}
void SubIOp::inferResultRanges(ArrayRef<ConstantIntRanges> ranges, SetIntRangeFn setResultRange) {
  setResultRange(getResult(), intrange::inferSub(ranges));
}
void MulIOp::inferResultRanges(ArrayRef<ConstantIntRanges> ranges, SetIntRangeFn setResultRange) {
  setResultRange(getResult(), intrange::inferMul(ranges));
}
void AndIOp::inferResultRanges(ArrayRef<ConstantIntRanges> ranges, SetIntRangeFn setResultRange) {
  setResultRange(getResult(), intrange::inferAnd(ranges));
}
void OrIOp::inferResultRanges(ArrayRef<ConstantIntRanges> ranges, SetIntRangeFn setResultRange) {
  setResultRange(getResult(), intrange::inferOr(ranges));
}
void XOrIOp::inferResultRanges(ArrayRef<ConstantIntRanges> ranges, SetIntRangeFn setResultRange) {
  setResultRange(getResult(), intrange::inferXor(ranges));
}
void EqOp::inferResultRanges(ArrayRef<ConstantIntRanges>, SetIntRangeFn setResultRange) {
  setResultRange(getResult(), ConstantIntRanges::fromUnsigned(APInt(1, 0), APInt(1, 1)));
}
void LiftOp::inferResultRanges(ArrayRef<ConstantIntRanges> ranges, SetIntRangeFn setResultRange) {
  setResultRange(getResult(), ranges[0]);
}
void InstanceOp::inferResultRanges(ArrayRef<ConstantIntRanges> ranges, SetIntRangeFn setResultRange) {
  setResultRange(getResult(), ranges[0]);
}
void AddAssignOp::inferResultRanges(ArrayRef<ConstantIntRanges> ranges, SetIntRangeFn setResultRange) {
  setResultRange(getResult(), intrange::inferAdd(ranges));
}
void SubAssignOp::inferResultRanges(ArrayRef<ConstantIntRanges> ranges, SetIntRangeFn setResultRange) {
  setResultRange(getResult(), intrange::inferSub(ranges));
}
void SltOp::inferResultRanges(ArrayRef<ConstantIntRanges>, SetIntRangeFn setResultRange) {
  setResultRange(getResult(), ConstantIntRanges::fromUnsigned(APInt(1, 0), APInt(1, 1)));
}
void SelectOp::inferResultRanges(ArrayRef<ConstantIntRanges> ranges, SetIntRangeFn setResultRange) {
  size_t n = getResults().size();  // the operands are the condition, then the n true values, then the n false values
  for (size_t i = 0; i < n; ++i)
    setResultRange(getResults()[i], ranges[1 + i].rangeUnion(ranges[1 + n + i]));
}
void MeasureOp::inferResultRanges(ArrayRef<ConstantIntRanges> ranges, SetIntRangeFn setResultRange) {
  setResultRange(getResult(), ranges[0]);
}

// ---- references ------------------------------------------------------------------------------------------------------

LogicalResult CondOp::verify() {
  if (!elementOf(getInput().getType()).isInteger(1))
    return emitOpError("needs a distribution of one-bit integers");
  return success();
}


// ---- folding definite distributions -----------------------------------------------------------------------------------------

// The integer a known operand stands for: a point (a definite distribution) or an integer constant.
static std::optional<APInt> known(Attribute attr) {
  if (auto point = dyn_cast_or_null<PointAttr>(attr))
    return point.getValue().getValue();
  if (auto integer = dyn_cast_or_null<IntegerAttr>(attr))
    return integer.getValue();
  return std::nullopt;
}
static Attribute pointOf(Type result, const APInt &value) {
  return PointAttr::get(result.getContext(), IntegerAttr::get(elementOf(result), value));
}

Operation *DistDialect::materializeConstant(OpBuilder &builder, Attribute value, Type type, Location loc) {
  if (auto point = dyn_cast<PointAttr>(value))
    return ConstantOp::create(builder, loc, type, point);
  return arith::ConstantOp::materialize(builder, value, type, loc);
}

OpFoldResult LiftOp::fold(FoldAdaptor adaptor) {
  if (auto value = known(adaptor.getInput()))
    return pointOf(getType(), *value);
  return {};
}
OpFoldResult InstanceOp::fold(FoldAdaptor adaptor) {  // an instance of a definite value is another definite value
  if (isa_and_nonnull<PointAttr>(adaptor.getInput()))
    return adaptor.getInput();
  return {};
}
OpFoldResult MeasureOp::fold(FoldAdaptor adaptor) {
  if (auto value = known(adaptor.getInput()))
    return IntegerAttr::get(getType(), value->sextOrTrunc(getType().getIntOrFloatBitWidth()));
  return {};
}
OpFoldResult CondOp::fold(FoldAdaptor adaptor) {
  if (auto value = known(adaptor.getInput()))
    return IntegerAttr::get(getType(), *value);
  return {};
}
OpFoldResult IsSingleOp::fold(FoldAdaptor adaptor) {
  if (known(adaptor.getInput()))
    return BoolAttr::get(getContext(), true);
  if (isa_and_nonnull<UnitAttr>(adaptor.getInput()))  // what constant propagation calls a superposition
    return BoolAttr::get(getContext(), false);
  return {};
}
template <typename Op> static OpFoldResult foldBinary(Op op, typename Op::FoldAdaptor adaptor, APInt (*compute)(const APInt &, const APInt &)) {
  auto lhs = known(adaptor.getOperands()[0]), rhs = known(adaptor.getOperands()[1]);
  if (lhs && rhs)
    return pointOf(op.getType(), compute(*lhs, *rhs));
  return {};
}
static APInt plus(const APInt &a, const APInt &b) { return a + b; }
static APInt minus(const APInt &a, const APInt &b) { return a - b; }
static APInt times(const APInt &a, const APInt &b) { return a * b; }
static APInt equal(const APInt &a, const APInt &b) { return APInt(1, a == b); }
static APInt both(const APInt &a, const APInt &b) { return a & b; }
static APInt either(const APInt &a, const APInt &b) { return a | b; }
static APInt differ(const APInt &a, const APInt &b) { return a ^ b; }
static APInt less(const APInt &a, const APInt &b) { return APInt(1, a.slt(b)); }
OpFoldResult AddIOp::fold(FoldAdaptor adaptor) { return foldBinary(*this, adaptor, plus); }
OpFoldResult SubIOp::fold(FoldAdaptor adaptor) { return foldBinary(*this, adaptor, minus); }
OpFoldResult SltOp::fold(FoldAdaptor adaptor) {
  if (getLhs() == getRhs())
    return pointOf(getType(), APInt(1, 0));
  return foldBinary(*this, adaptor, less);
}
OpFoldResult MulIOp::fold(FoldAdaptor adaptor) { return foldBinary(*this, adaptor, times); }
// An operation on a value and itself is one roll used twice, so it is known without knowing the value: x && x and x || x are x,
// x ^ x and x < x are 0, x == x is 1.
OpFoldResult EqOp::fold(FoldAdaptor adaptor) {
  if (getLhs() == getRhs())
    return pointOf(getType(), APInt(1, 1));
  return foldBinary(*this, adaptor, equal);
}
OpFoldResult AndIOp::fold(FoldAdaptor adaptor) { return getLhs() == getRhs() ? OpFoldResult(getLhs()) : foldBinary(*this, adaptor, both); }
OpFoldResult OrIOp::fold(FoldAdaptor adaptor) { return getLhs() == getRhs() ? OpFoldResult(getLhs()) : foldBinary(*this, adaptor, either); }
OpFoldResult XOrIOp::fold(FoldAdaptor adaptor) {
  if (getLhs() == getRhs())
    return pointOf(getType(), APInt(elementOf(getType()).getIntOrFloatBitWidth(), 0));
  return foldBinary(*this, adaptor, differ);
}
OpFoldResult AddAssignOp::fold(FoldAdaptor adaptor) { return foldBinary(*this, adaptor, plus); }
OpFoldResult SubAssignOp::fold(FoldAdaptor adaptor) { return foldBinary(*this, adaptor, minus); }
OpFoldResult ConstantOp::fold(FoldAdaptor) { return getValue(); }
void ConstantOp::inferResultRanges(ArrayRef<ConstantIntRanges>, SetIntRangeFn setResultRange) {
  APInt value = getValue().getValue().getValue();
  setResultRange(getResult(), ConstantIntRanges::constant(value));
}

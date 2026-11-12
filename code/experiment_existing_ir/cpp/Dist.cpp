#include "Dialects.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
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

LogicalResult MarkOp::verify() {
  if (!elementOf(getPredicate().getType()).isInteger(1))
    return emitOpError("needs a predicate that is a distribution of one-bit integers");
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
// A comparison is known when the operands' ranges decide it, as for arith.cmpi: `a < 10` of a value in 0..3 is always true.
static ConstantIntRanges rangeOfComparison(intrange::CmpPredicate predicate, ArrayRef<ConstantIntRanges> ranges) {
  if (std::optional<bool> decided = intrange::evaluatePred(predicate, ranges[0], ranges[1]))
    return ConstantIntRanges::constant(APInt(1, *decided));
  return ConstantIntRanges::fromUnsigned(APInt(1, 0), APInt(1, 1));
}
void EqOp::inferResultRanges(ArrayRef<ConstantIntRanges> ranges, SetIntRangeFn setResultRange) {
  setResultRange(getResult(), rangeOfComparison(intrange::CmpPredicate::eq, ranges));
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
void SltOp::inferResultRanges(ArrayRef<ConstantIntRanges> ranges, SetIntRangeFn setResultRange) {
  setResultRange(getResult(), rangeOfComparison(intrange::CmpPredicate::slt, ranges));
}
void SelectOp::inferResultRanges(ArrayRef<ConstantIntRanges> ranges, SetIntRangeFn setResultRange) {
  size_t n = getResults().size();  // the operands are the condition, then the n true values, then the n false values
  for (size_t i = 0; i < n; ++i)
    setResultRange(getResults()[i], ranges[1 + i].rangeUnion(ranges[1 + n + i]));
}
void ReflectOp::inferResultRanges(ArrayRef<ConstantIntRanges> ranges, SetIntRangeFn setResultRange) {
  setResultRange(getResult(), ranges[0]);  // the weights move between outcomes, none to a value it could not have
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
// An update by a known zero is the target again: the result of an update is the target, so nothing is lost.
template <typename Op> static OpFoldResult foldStep(Op op, typename Op::FoldAdaptor adaptor, APInt (*compute)(const APInt &, const APInt &)) {
  std::optional<APInt> amount = known(adaptor.getOperands()[1]);
  if (amount && amount->isZero() && isa<DistType>(op->getOperand(0).getType()))
    return op->getOperand(0);
  return foldBinary(op, adaptor, compute);
}
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
// ---- combining constant steps -----------------------------------------------------------------------------------------

// A sum or difference of a constant taken from another one with a constant is one step: (x + 1) + 2 is x + 3, as arith does.
// The inner step must have no other user; its result is only the outer step's operand. `y = a + 1; z = y + 1` is such a chain
// (each assignment replays `a`, so the chain is on one value), and one step costs one adder where the chain cost several.
template <typename Op> struct CombineConstantSteps : OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  static bool isStep(Operation *op) { return isa<AddIOp, SubIOp, AddAssignOp, SubAssignOp>(op); }
  static bool subtracts(Operation *op) { return isa<SubIOp, SubAssignOp>(op); }
  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const override {
    Operation *inner = op->getOperand(0).getDefiningOp();
    APInt outerAmount, innerAmount;
    if (!inner || !isStep(inner) || !inner->getResult(0).hasOneUse() ||
        !matchPattern(op->getOperand(1), m_ConstantInt(&outerAmount)) || !matchPattern(inner->getOperand(1), m_ConstantInt(&innerAmount)) ||
        outerAmount.getBitWidth() != innerAmount.getBitWidth())
      return failure();
    APInt total = (subtracts(inner) ? -innerAmount : innerAmount) + (subtracts(op) ? -outerAmount : outerAmount);
    if (total.isZero())
      return failure();
    bool inPlace = bool(existing_ir::consumedBy(inner));  // the inner step consumed its operand, so the combined one does
    Value operand = inner->getOperand(0);
    Value amount = arith::ConstantOp::create(rewriter, op->getLoc(), rewriter.getIntegerAttr(op->getOperand(1).getType(), total.isNegative() ? -total : total));
    Type type = op->getResult(0).getType();
    Operation *step;
    if (total.isNegative())
      step = inPlace ? SubAssignOp::create(rewriter, op->getLoc(), type, operand, amount).getOperation()
                     : SubIOp::create(rewriter, op->getLoc(), type, operand, amount).getOperation();
    else
      step = inPlace ? AddAssignOp::create(rewriter, op->getLoc(), type, operand, amount).getOperation()
                     : AddIOp::create(rewriter, op->getLoc(), type, operand, amount).getOperation();
    rewriter.replaceOp(op, step->getResults());
    return success();
  }
};

// ---- flattening nested conditionals ------------------------------------------------------------------------------------------

// A conditional inside a conditional that otherwise gives the same value takes one condition: if c { if d { x } else { y } } else
// { y } is `x` where c && d and `y` elsewhere, and likewise with the roles of the arms swapped for ||. A nest of n conditions
// becomes one condition of n - 1 `and`s, which the lowering computes without a select and a copy of the value at each level.
// Only a select of one value is flattened; the inner select must have no other user.
struct FlattenSelect : OpRewritePattern<SelectOp> {
  using OpRewritePattern<SelectOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SelectOp outer, PatternRewriter &rewriter) const override {
    if (outer.getResults().size() != 1)
      return failure();
    Value whenTrue = outer.getTrueValues()[0], whenFalse = outer.getFalseValues()[0];
    for (bool nestedInTrue : {true, false}) {
      auto inner = (nestedInTrue ? whenTrue : whenFalse).getDefiningOp<SelectOp>();
      if (!inner || inner.getResults().size() != 1 || !inner->hasOneUse())
        continue;
      Value same = nestedInTrue ? whenFalse : whenTrue;  // what the outer gives when the inner is not reached
      Value innerTrue = inner.getTrueValues()[0], innerFalse = inner.getFalseValues()[0];
      Type type = outer.getCondition().getType();
      if (nestedInTrue && innerFalse == same) {
        Value both = AndIOp::create(rewriter, outer.getLoc(), type, outer.getCondition(), inner.getCondition());
        rewriter.replaceOpWithNewOp<SelectOp>(outer, TypeRange{outer.getResult(0).getType()}, both, ValueRange{innerTrue}, ValueRange{same});
        return success();
      }
      if (!nestedInTrue && innerTrue == same) {
        Value either = OrIOp::create(rewriter, outer.getLoc(), type, outer.getCondition(), inner.getCondition());
        rewriter.replaceOpWithNewOp<SelectOp>(outer, TypeRange{outer.getResult(0).getType()}, either, ValueRange{same}, ValueRange{innerFalse});
        return success();
      }
    }
    return failure();
  }
};
// `x + 0` and `x - 0` computed out of place are a copy of x that can be updated without touching x, and that is kept when it is
// needed: when x has other uses and the copy is updated in place. Where the copy is only read, or x has no other use and may as
// well be the one updated, it is x.
template <typename Op> struct DropZeroStep : OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const override {
    Attribute amount;
    Value operand = op->getOperand(0);
    if (!isa<DistType>(operand.getType()) || !matchPattern(op->getOperand(1), m_Constant(&amount)))
      return failure();
    std::optional<APInt> value = known(amount);
    if (!value || !value->isZero())
      return failure();
    bool updated = llvm::any_of(op->getUsers(), [&](Operation *user) { return existing_ir::consumes(user, op->getResult(0)); });
    if (updated && !operand.hasOneUse())
      return failure();
    rewriter.replaceOp(op, operand);
    return success();
  }
};

void existing_ir::populateDistRewrites(RewritePatternSet &patterns, bool combineSteps, bool flatten) {
  if (combineSteps)
    patterns.add<DropZeroStep<AddIOp>, DropZeroStep<SubIOp>, CombineConstantSteps<AddIOp>, CombineConstantSteps<SubIOp>, CombineConstantSteps<AddAssignOp>,
                 CombineConstantSteps<SubAssignOp>>(patterns.getContext());
  if (flatten)
    patterns.add<FlattenSelect>(patterns.getContext());
}

OpFoldResult AddAssignOp::fold(FoldAdaptor adaptor) { return foldStep(*this, adaptor, plus); }
OpFoldResult SubAssignOp::fold(FoldAdaptor adaptor) { return foldStep(*this, adaptor, minus); }
OpFoldResult ConstantOp::fold(FoldAdaptor) { return getValue(); }
void ConstantOp::inferResultRanges(ArrayRef<ConstantIntRanges>, SetIntRangeFn setResultRange) {
  APInt value = getValue().getValue().getValue();
  setResultRange(getResult(), ConstantIntRanges::constant(value));
}

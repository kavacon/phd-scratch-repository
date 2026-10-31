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
  if (getStop() <= getStart())
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
LogicalResult SltOp::verify() { return verifyBinary(*this, true); }

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
  setResultRange(getResult(), ConstantIntRanges::fromSigned(APInt(width, getStart(), true), APInt(width, getStop() - 1, true)));
}
void AddIOp::inferResultRanges(ArrayRef<ConstantIntRanges> ranges, SetIntRangeFn setResultRange) {
  setResultRange(getResult(), intrange::inferAdd(ranges));
}
void SubIOp::inferResultRanges(ArrayRef<ConstantIntRanges> ranges, SetIntRangeFn setResultRange) {
  setResultRange(getResult(), intrange::inferSub(ranges));
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

// ---- objects -------------------------------------------------------------------------------------------------------

// A reference and the distribution it holds have the same element type.
static LogicalResult verifyHolds(Operation *op, Type ref, Type value) {
  if (cast<RefType>(ref).getElement() != elementOf(value))
    return op->emitOpError("needs a reference whose element type is the distribution's");
  return success();
}
LogicalResult NewOp::verify() { return verifyHolds(*this, getResult().getType(), getInitial().getType()); }
LogicalResult GetOp::verify() { return verifyHolds(*this, getRef().getType(), getResult().getType()); }
LogicalResult SetOp::verify() { return verifyHolds(*this, getRef().getType(), getValue().getType()); }
LogicalResult OverwriteOp::verify() { return verifyHolds(*this, getRef().getType(), getValue().getType()); }
LogicalResult ShareOp::verify() {
  if (getSource().getType() != getResult().getType())
    return emitOpError("needs a result of the type of its source");
  return success();
}
LogicalResult AssignOp::verify() {
  if (getTarget().getType() != getSource().getType() || getTarget().getType() != getResult().getType())
    return emitOpError("needs references of one type");
  return success();
}
LogicalResult CondOp::verify() {
  if (!elementOf(getInput().getType()).isInteger(1))
    return emitOpError("needs a distribution of one-bit integers");
  return success();
}

// A condition that is a lift of an integer is that integer.
OpFoldResult CondOp::fold(FoldAdaptor) {
  if (auto lift = getInput().getDefiningOp<LiftOp>())
    return lift.getInput();
  return {};
}

// Promoting an object to a value: `mem2reg` treats `new` as an allocation whose first content is its initial value, `get` as
// a load and `set` as a store. It does so only when nothing else uses the reference (a marker, or a yield out of a
// conditional), so an object that is still ambiguous stays in memory.
SmallVector<MemorySlot> NewOp::getPromotableSlots() { return {MemorySlot{getResult(), getInitial().getType()}}; }
Value NewOp::getDefaultValue(const MemorySlot &, OpBuilder &) { return getInitial(); }
void NewOp::handleBlockArgument(const MemorySlot &, BlockArgument, OpBuilder &) {}
std::optional<PromotableAllocationOpInterface> NewOp::handlePromotionComplete(const MemorySlot &, Value, OpBuilder &) {
  erase();
  return std::nullopt;
}

// The reference is the only thing a memory operation may block with, and as its operand.
static bool usesOnlyTheSlot(Value ref, const SmallPtrSetImpl<OpOperand *> &blocking, const MemorySlot &slot) {
  return blocking.size() == 1 && (*blocking.begin())->get() == slot.ptr && ref == slot.ptr;
}
bool GetOp::loadsFrom(const MemorySlot &slot) { return getRef() == slot.ptr; }
bool GetOp::storesTo(const MemorySlot &) { return false; }
Value GetOp::getStored(const MemorySlot &, OpBuilder &, Value, const DataLayout &) { return {}; }
bool GetOp::canUsesBeRemoved(const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blocking,
                             SmallVectorImpl<OpOperand *> &, const DataLayout &) {
  return usesOnlyTheSlot(getRef(), blocking, slot);
}
DeletionKind GetOp::removeBlockingUses(const MemorySlot &, const SmallPtrSetImpl<OpOperand *> &, OpBuilder &,
                                       Value reachingDefinition, const DataLayout &) {
  getResult().replaceAllUsesWith(reachingDefinition);
  return DeletionKind::Delete;
}
bool SetOp::loadsFrom(const MemorySlot &) { return false; }
bool SetOp::storesTo(const MemorySlot &slot) { return getRef() == slot.ptr; }
Value SetOp::getStored(const MemorySlot &, OpBuilder &, Value, const DataLayout &) { return getValue(); }
bool SetOp::canUsesBeRemoved(const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blocking,
                             SmallVectorImpl<OpOperand *> &, const DataLayout &) {
  return usesOnlyTheSlot(getRef(), blocking, slot);
}
DeletionKind SetOp::removeBlockingUses(const MemorySlot &, const SmallPtrSetImpl<OpOperand *> &, OpBuilder &, Value,
                                       const DataLayout &) {
  return DeletionKind::Delete;
}

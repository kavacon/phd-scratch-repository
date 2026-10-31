// Canonicalizing the operations on lifted integers.
//
// A variable that may hold a distribution holds a lifted integer (`dist.lift`) on the paths where nothing superposes it, and
// the operations on it are distribution operations that `arith` would have folded. These patterns, part of `canonicalize`,
// rewrite every operation whose distribution operands are all lifted integers back into `arith` on the integers, lifting
// only the result, so a definite value is a lift of an integer (which `kindOf` in Refs.cpp relies on) and constant
// arithmetic folds. A lift of a value that is later made superposed is not touched: one of its uses has an operand that is
// not a lift.
#include "Dialects.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace existing_ir::dist;

namespace {

// The integer a value stands for when it is definite: an integer, or the integer a lift was made from.
Value definite(Value value) {
  if (isa<IntegerType>(value.getType()))
    return value;
  if (auto lift = value.getDefiningOp<LiftOp>())
    return lift.getInput();
  return {};
}

// A sum, difference or update of definite operands is the lift of the integer arithmetic.
template <typename Op, typename ArithOp>
struct Arithmetic : OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op, PatternRewriter &rw) const override {
    Value lhs = definite(op->getOperand(0)), rhs = definite(op->getOperand(1));
    if (!lhs || !rhs)
      return failure();
    Value result = ArithOp::create(rw, op.getLoc(), lhs, rhs);
    rw.replaceOpWithNewOp<LiftOp>(op, op->getResult(0).getType(), result);
    return success();
  }
};

struct Compare : OpRewritePattern<SltOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SltOp op, PatternRewriter &rw) const override {
    Value lhs = definite(op.getLhs()), rhs = definite(op.getRhs());
    if (!lhs || !rhs)
      return failure();
    Value result = arith::CmpIOp::create(rw, op.getLoc(), arith::CmpIPredicate::slt, lhs, rhs);
    rw.replaceOpWithNewOp<LiftOp>(op, op.getResult().getType(), result);
    return success();
  }
};

// An instance of a definite value is another definite value.
struct Instance : OpRewritePattern<InstanceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(InstanceOp op, PatternRewriter &rw) const override {
    Value input = definite(op.getInput());
    if (!input)
      return failure();
    rw.replaceOpWithNewOp<LiftOp>(op, op.getResult().getType(), input);
    return success();
  }
};

// Measuring a definite value is the value.
struct Measure : OpRewritePattern<MeasureOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MeasureOp op, PatternRewriter &rw) const override {
    Value input = definite(op.getInput());
    if (!input)
      return failure();
    rw.replaceOp(op, input);
    return success();
  }
};

// A select on a definite condition takes one side. With a constant condition that is a choice of operand; otherwise it is
// an integer select when both sides are definite.
struct Select : OpRewritePattern<SelectOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SelectOp op, PatternRewriter &rw) const override {
    Value condition = definite(op.getCondition());
    if (!condition)
      return failure();
    SmallVector<Value> results;
    for (auto [t, f, r] : llvm::zip(op.getTrueValues(), op.getFalseValues(), op.getResults())) {
      Value chosen;
      if (matchPattern(condition, m_One()))
        chosen = t;
      else if (matchPattern(condition, m_Zero()))
        chosen = f;
      if (chosen) {
        if (isa<IntegerType>(chosen.getType()))
          chosen = LiftOp::create(rw, op.getLoc(), r.getType(), chosen);
        results.push_back(chosen);
        continue;
      }
      Value tt = definite(t), ff = definite(f);
      if (!tt || !ff)
        return failure();
      results.push_back(LiftOp::create(rw, op.getLoc(), r.getType(), arith::SelectOp::create(rw, op.getLoc(), condition, tt, ff)));
    }
    rw.replaceOp(op, results);
    return success();
  }
};

} // namespace

void AddIOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<Arithmetic<AddIOp, arith::AddIOp>>(context);
}
void SubIOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<Arithmetic<SubIOp, arith::SubIOp>>(context);
}
void AddAssignOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<Arithmetic<AddAssignOp, arith::AddIOp>>(context);
}
void SubAssignOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<Arithmetic<SubAssignOp, arith::SubIOp>>(context);
}
void SltOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<Compare>(context);
}
void InstanceOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<Instance>(context);
}
void MeasureOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<Measure>(context);
}
void SelectOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<Select>(context);
}

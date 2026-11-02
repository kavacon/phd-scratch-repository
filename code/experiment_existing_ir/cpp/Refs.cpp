// What is left of references and conditionals on a superposition once stock passes have done what they can.
//
// A variable that may hold a distribution holds a reference to an object: a `memref<!dist.dist<T>>` cell, read and written
// with `dist.get` and `dist.set`. `mem2reg` promotes the variable's slot (a memref of cells), which makes rebinding
// a name a value that branches yield. The cells stay in memory because `mem2reg` does not touch affine operations, and
// the stock `affine-scalrep` forwards a stored value to the reads that follow it and removes the stores nothing reads, so
// the objects become values once nothing says otherwise. `y = x` is `arith.select(is_single(x's value), copy, x)`, which
// folds when the value is known, and `canonicalize` finishes the rest.
//
// What this file adds to that is the conditional on a superposition. An `scf.if` on a `dist.cond` that stays has the objects
// its arms store to in memory. It becomes both arms, each working on a copy of those objects (so the arms do not meet), and
// one `dist.merge` per object, which is a select tying the object to the condition unless neither arm changed it.
#include "Dialects.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace existing_ir::dist;

namespace existing_ir::dist {

// A lift is definite and a range is not; an operation is a superposition if any operand is, and is otherwise not yet known
// (it becomes a lift when `canonicalize` rewrites it, see Recover.cpp).
Kind kindOf(Value value) {
  if (isa<IntegerType>(value.getType()))
    return Kind::Definite;
  Operation *op = value.getDefiningOp();
  if (!op)
    return Kind::Unknown;
  if (isa<LiftOp>(op))
    return Kind::Definite;
  if (isa<UniformOp>(op))
    return Kind::Superposed;
  if (auto select = dyn_cast<SelectOp>(op))  // tied to its condition whatever the arms hold
    return kindOf(select.getCondition()) == Kind::Superposed ? Kind::Superposed : Kind::Unknown;
  if (isa<AddIOp, SubIOp, SltOp, AddAssignOp, SubAssignOp, InstanceOp>(op))
    for (Value operand : op->getOperands())
      if (kindOf(operand) == Kind::Superposed)
        return Kind::Superposed;
  return Kind::Unknown;
}

bool same(Value a, Value b) {
  if (a == b)
    return true;
  auto left = a.getDefiningOp<LiftOp>(), right = b.getDefiningOp<LiftOp>();
  APInt x, y;
  return left && right && matchPattern(left.getInput(), m_ConstantInt(&x)) &&
         matchPattern(right.getInput(), m_ConstantInt(&y)) && x == y;
}

bool settled(Value value) {
  if (kindOf(value) == Kind::Superposed)
    return true;
  auto lift = value.getDefiningOp<LiftOp>();
  return lift && matchPattern(lift.getInput(), m_Constant());
}

} // namespace existing_ir::dist

namespace {

// An operation that may be done in both arms of a conditional: it only allocates, reads or has no effect.
bool speculatable(Operation *op) {
  if (isa<cf::AssertOp>(op))
    return true;
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  if (op->getNumRegions() != 0 || !effects)
    return false;
  SmallVector<MemoryEffects::EffectInstance> list;
  effects.getEffects(list);
  return llvm::all_of(list, [](auto &effect) { return !isa<MemoryEffects::Write>(effect.getEffect()); });
}

Value memrefOf(Operation *op) {
  if (auto load = dyn_cast<GetOp>(op))
    return load.getCell();
  if (auto store = dyn_cast<SetOp>(op))
    return store.getCell();
  return {};
}

// Two cells from different allocations are different objects. Anything else (a cell picked by a select) may be the same one.
bool distinct(Value a, Value b) { return a != b && a.getDefiningOp<memref::AllocaOp>() && b.getDefiningOp<memref::AllocaOp>(); }

// The value a cell holds just before `at`: the nearest store before it, looking back through straight-line code and out of the
// arms of `scf.if` (an arm starts with what held before the conditional). It stops, and gives nothing, at anything that may
// write the cell. `affine-scalrep` does this for a read in the block of the store, but not out of an `scf.if` arm, which is
// where the conditions of nested conditionals are.
Value reaching(Value cell, Operation *at) {
  for (;;) {
    for (Operation *op = at->getPrevNode(); op; op = op->getPrevNode()) {
      if (auto store = dyn_cast<SetOp>(op); store && store.getCell() == cell)
        return store.getValue();
      auto effects = getEffectsRecursively(op);
      if (!effects)
        return {};
      for (const MemoryEffects::EffectInstance &effect : *effects) {
        if (!isa<MemoryEffects::Write>(effect.getEffect()))
          continue;
        Value written = effect.getValue();
        if (!written || (isa<MemRefType>(written.getType()) && !distinct(written, cell)))
          return {};
      }
    }
    at = at->getParentOp();
    if (!isa<scf::IfOp>(at))
      return {};
  }
}

// Reading a cell after a store to it is the stored value.
struct ForwardGet : OpRewritePattern<GetOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GetOp op, PatternRewriter &rewriter) const override {
    Value held = reaching(op.getCell(), op);
    if (!held)
      return failure();
    rewriter.replaceOp(op, held);
    return success();
  }
};

// A conditional on a superposition, as a pattern of the dialect: `canonicalize` applies it. It does nothing for a conditional
// that is not yet certainly on a superposition, or whose arms touch a cell it cannot tell from another (one picked by a
// select that has not folded) or hold anything but straight-line code: it matches again after they are resolved.
struct FlattenDistIf : OpRewritePattern<scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::IfOp op, PatternRewriter &rewriter) const override {
    auto condition = op.getCondition().getDefiningOp<CondOp>();
    if (!condition || kindOf(condition.getInput()) != Kind::Superposed)
      return failure();
    SmallVector<Value> changed;  // the cells from outside the conditional that an arm stores to
    for (Region *region : {&op.getThenRegion(), &op.getElseRegion()}) {
      if (region->empty())
        continue;
      for (Operation &inner : region->front().without_terminator()) {
        if (Value cell = memrefOf(&inner)) {
          if (!cell.getDefiningOp<memref::AllocaOp>())  // a cell picked by a select may be any of its operands
            return failure();
          if (isa<SetOp>(inner) && !op->isAncestor(cell.getParentBlock()->getParentOp()) && !llvm::is_contained(changed, cell))
            changed.push_back(cell);
        } else if (!speculatable(&inner)) {
          return failure();
        }
      }
    }
    // A name rebound in the arms is bound to different objects in them, which would depend on the outcome: left for
    // `check-resolved` to refuse.
    for (auto [index, result] : llvm::enumerate(op.getResults()))
      if (op.getElseRegion().empty() || op.thenYield().getOperand(index) != op.elseYield().getOperand(index))
        return failure();
    Location loc = op.getLoc();
    SmallVector<Value> before, shadows[2];
    for (Value cell : changed) {  // each arm works on a copy of the cell, starting from what it holds
      before.push_back(GetOp::create(rewriter, loc, cast<MemRefType>(cell.getType()).getElementType(), cell));
      for (int arm = 0; arm < 2; ++arm) {
        Value copy = memref::AllocaOp::create(rewriter, loc, cast<MemRefType>(cell.getType()));
        SetOp::create(rewriter, loc, before.back(), copy);
        shadows[arm].push_back(copy);
        Region *region = arm == 0 ? &op.getThenRegion() : &op.getElseRegion();
        rewriter.replaceUsesWithIf(cell, copy, [&](OpOperand &use) { return !isa<scf::YieldOp>(use.getOwner()) && region->isAncestor(use.getOwner()->getParentRegion()); });
      }
    }
    for (auto [index, result] : llvm::enumerate(op.getResults()))
      rewriter.replaceAllUsesWith(result, op.thenYield().getOperand(index));
    for (Region *region : {&op.getThenRegion(), &op.getElseRegion()})
      if (!region->empty())
        for (Operation &inner : llvm::make_early_inc_range(region->front().without_terminator()))
          rewriter.moveOpBefore(&inner, op);
    for (auto [index, cell] : llvm::enumerate(changed)) {
      Value whenTrue = GetOp::create(rewriter, loc, before[index].getType(), shadows[0][index]);
      Value whenFalse = GetOp::create(rewriter, loc, before[index].getType(), shadows[1][index]);
      Value merged = MergeOp::create(rewriter, loc, before[index].getType(), condition.getInput(), before[index], whenTrue, whenFalse);
      SetOp::create(rewriter, loc, merged, cell);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// A cell that is only stored to: `affine-scalrep` forwards the reads away but leaves the stores, and the memref dialect's own
// removal of such an allocation only knows `memref.store`.
struct DeadCell : OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::AllocaOp op, PatternRewriter &rewriter) const override {
    if (!isa<DistType>(op.getType().getElementType()) ||
        !llvm::all_of(op->getUsers(), [](Operation *user) { return isa<SetOp>(user); }))
      return failure();
    for (Operation *user : llvm::make_early_inc_range(op->getUsers()))
      rewriter.eraseOp(user);
    rewriter.eraseOp(op);
    return success();
  }
};

// A merge whose arms changed the object: a select, once the values can be told apart.
struct SelectMerge : OpRewritePattern<MergeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MergeOp op, PatternRewriter &rewriter) const override {
    if (!settled(op.getBefore()) || !settled(op.getWhenTrue()) || !settled(op.getWhenFalse()))
      return failure();
    rewriter.replaceOpWithNewOp<SelectOp>(op, TypeRange{op.getType()}, op.getCondition(), ValueRange{op.getWhenTrue()},
                                          ValueRange{op.getWhenFalse()});
    return success();
  }
};

// Whatever is left after the pipeline has folded what it can is a program the semantics refuse (a drop, a name bound to
// different objects under a condition on a distribution) or one the translation cannot decide.
struct CheckResolved : PassWrapper<CheckResolved, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CheckResolved)
  StringRef getArgument() const final { return "check-resolved"; }
  StringRef getDescription() const final { return "Report the references, checks and conditionals that were not resolved"; }
  void getDependentDialects(DialectRegistry &registry) const override { registry.insert<DistDialect>(); }

  void runOnOperation() override {
    getOperation().walk([&](Operation *op) {
      if (auto check = dyn_cast<cf::AssertOp>(op)) {
        op->emitError(check.getMsg());
        return signalPassFailure();
      }
      bool object = llvm::any_of(op->getResultTypes(), [](Type t) {
        auto memref = dyn_cast<MemRefType>(t);
        return memref && (isa<DistType>(memref.getElementType()) || isa<MemRefType>(memref.getElementType()));
      });
      if (object && isa<arith::SelectOp, scf::IfOp>(op))
        op->emitError("which object a name means could not be decided: it is bound to different objects in the arms of a "
                      "conditional on a distribution, or depends on something that is not known when compiling");
      else if (object || isa<GetOp, SetOp, CondOp, IsSingleOp, MergeOp>(op))
        op->emitError("could not be resolved: what this name refers to depends on something that is not known when "
                      "compiling");
      else
        return;
      signalPassFailure();
    });
  }
};

} // namespace

void GetOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<ForwardGet>(context);
}
void MergeOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<SelectMerge>(context);
}
void DistDialect::getCanonicalizationPatterns(RewritePatternSet &results) const {
  results.add<FlattenDistIf, DeadCell>(getContext());
}

void existing_ir::registerCheckResolved() { PassRegistration<CheckResolved>(); }

// References to objects, and the conditionals on a superposition.
//
// A name that can hold a distribution holds a reference to an object (`dist.new`, `dist.get`, `dist.set`). `mem2reg`
// promotes an object to a value, but only when nothing but loads and stores uses its reference, and a reference that is
// yielded out of a conditional or shared under a second name is such a use. What those uses mean depends on the value the
// object holds, so each is a marker (`dist.share`, `dist.assign`, `dist.overwrite`) that a canonicalization pattern here
// replaces once that value is known. The value is found by looking back along the straight-line code (`reaching`), which is
// all `canonicalize` has to work with: it is the part of store-to-load forwarding that `mem2reg` cannot do for an object
// whose reference escapes. Folding a condition needs the same lookup, so `dist.get` forwards too.
//
// Whether a value is definite (an integer, so a copy is a copy) or a superposition (so a copy is an alias) is read off the
// operation that made it (`kindOf`), after the patterns of `canonicalize` (Recover.cpp) have turned arithmetic on lifted integers back into lifts.
// Until the kind is certain a marker stays, and `resolve-refs` repeats `canonicalize`, `mem2reg` and `flatten-dist-if` until nothing changes.
//
// `flatten-dist-if` is the other half. A conditional on a superposition (an `scf.if` on a `dist.cond` that stays) holds the
// objects its arms change in memory, because the condition names them (`touched`). When the arms are straight-line and the
// value each changed object holds at the end of each arm is known, the conditional is both arms computed out of place, and
// each changed object is stored `select(condition, then value, else value)`: the variable is tied to the condition. An
// object an arm leaves as it was (a store of an equal integer is that too) is not changed, as the semantics say.
#include "Dialects.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
using namespace existing_ir::dist;

namespace {

enum class Kind { Definite, Superposed, Unknown };

// What a value of an object is, from how it was made. A lift is definite and a range is not; an operation is a
// superposition if any operand is, and is otherwise not yet known (it becomes a lift when `canonicalize` rewrites it).
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

// Two references to different allocations are different objects. Anything else may be the same one.
bool distinct(Value a, Value b) { return a != b && a.getDefiningOp<NewOp>() && b.getDefiningOp<NewOp>(); }

// The value an object holds just before `at`: the nearest store or creation before it, looking back through straight-line
// code and out of the arms of `scf.if` (the arm starts with what held before the conditional). It stops, and gives nothing,
// at anything that may write the object.
Value reaching(Value ref, Operation *at) {
  for (;;) {
    for (Operation *op = at->getPrevNode(); op; op = op->getPrevNode()) {
      if (auto set = dyn_cast<SetOp>(op); set && set.getRef() == ref)
        return set.getValue();
      if (auto created = dyn_cast<NewOp>(op); created && created.getResult() == ref)
        return created.getInitial();
      auto effects = getEffectsRecursively(op);
      if (!effects)
        return {};
      for (const MemoryEffects::EffectInstance &effect : *effects) {
        if (!isa<MemoryEffects::Write>(effect.getEffect()))
          continue;
        Value written = effect.getValue();
        if (!written)
          return {};
        if (isa<RefType>(written.getType()) && !distinct(written, ref))
          return {};
      }
    }
    at = at->getParentOp();
    if (!isa<scf::IfOp>(at))
      return {};
  }
}

Value copyOf(PatternRewriter &rewriter, Location loc, Value definite) {
  return LiftOp::create(rewriter, loc, definite.getType(), definite.getDefiningOp<LiftOp>().getInput());
}

// Reading an object after a store to it is the stored value.
struct ForwardGet : OpRewritePattern<GetOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GetOp op, PatternRewriter &rewriter) const override {
    Value held = reaching(op.getRef(), op);
    if (!held)
      return failure();
    rewriter.replaceOp(op, held);
    return success();
  }
};

// `y = x` with a new y: a new object when x holds an integer, x itself when it holds a superposition.
struct ResolveShare : OpRewritePattern<ShareOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ShareOp op, PatternRewriter &rewriter) const override {
    Value held = reaching(op.getSource(), op);
    if (!held)
      return failure();
    switch (kindOf(held)) {
    case Kind::Definite:
      rewriter.replaceOpWithNewOp<NewOp>(op, op.getType(), copyOf(rewriter, op.getLoc(), held));
      return success();
    case Kind::Superposed:
      rewriter.replaceOp(op, op.getSource());
      return success();
    default:
      return failure();
    }
  }
};

// `y = x` with an existing y, which must hold an integer: a store of the integer, or x itself.
struct ResolveAssign : OpRewritePattern<AssignOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AssignOp op, PatternRewriter &rewriter) const override {
    Value target = reaching(op.getTarget(), op), source = reaching(op.getSource(), op);
    if (!target || !source || kindOf(target) != Kind::Definite)
      return failure();
    switch (kindOf(source)) {
    case Kind::Definite:
      SetOp::create(rewriter, op.getLoc(), op.getTarget(), copyOf(rewriter, op.getLoc(), source));
      rewriter.replaceOp(op, op.getTarget());
      return success();
    case Kind::Superposed:
      rewriter.replaceOp(op, op.getSource());
      return success();
    default:
      return failure();
    }
  }
};

// `y = e` with an existing y: a store, unless y holds a distribution, which stays for `check-resolved` to refuse.
struct ResolveOverwrite : OpRewritePattern<OverwriteOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(OverwriteOp op, PatternRewriter &rewriter) const override {
    Value held = reaching(op.getRef(), op);
    if (!held || kindOf(held) != Kind::Definite)
      return failure();
    rewriter.replaceOpWithNewOp<SetOp>(op, op.getRef(), op.getValue());
    return success();
  }
};

// An operation that only allocates, or has no effects, and so may be done in both arms of a conditional.
bool speculatable(Operation *op) {
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  if (op->getNumRegions() != 0 || !effects)
    return false;
  SmallVector<MemoryEffects::EffectInstance> list;
  effects.getEffects(list);
  return llvm::all_of(list, [](auto &effect) { return isa<MemoryEffects::Allocate>(effect.getEffect()); });
}

// Whether two values are the same: one value, or two integers that are equal. (A superposition is only ever equal to itself.)
bool same(Value a, Value b) {
  if (a == b)
    return true;
  auto left = a.getDefiningOp<LiftOp>(), right = b.getDefiningOp<LiftOp>();
  APInt x, y;
  return left && right && matchPattern(left.getInput(), m_ConstantInt(&x)) &&
         matchPattern(right.getInput(), m_ConstantInt(&y)) && x == y;
}

// Whether a value is settled enough to tell from another: a superposition, or an integer that has folded to a constant.
bool settled(Value value) {
  if (kindOf(value) == Kind::Superposed)
    return true;
  auto lift = value.getDefiningOp<LiftOp>();
  return lift && matchPattern(lift.getInput(), m_Constant());
}

struct FlattenDistIf : PassWrapper<FlattenDistIf, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FlattenDistIf)
  StringRef getArgument() const final { return "flatten-dist-if"; }
  StringRef getDescription() const final {
    return "Turn an scf.if on a superposition into both arms and a dist.select of what they change";
  }
  void getDependentDialects(DialectRegistry &registry) const override { registry.insert<DistDialect>(); }

  // Does nothing for a conditional that is not yet certainly on a superposition, or whose arms still hold anything but
  // allocations and stores (reads that did not forward, markers, conditionals of their own): the pipeline tries again after
  // they are resolved. None of that depends on this conditional, so waiting cannot deadlock.
  LogicalResult flatten(scf::IfOp op) {
    auto condition = op.getCondition().getDefiningOp<CondOp>();
    if (!condition || kindOf(condition.getInput()) != Kind::Superposed)
      return success();
    SmallVector<Value> changed;  // the objects from outside the conditional that an arm stores to
    SmallVector<SetOp> stores;
    for (Region *region : {&op.getThenRegion(), &op.getElseRegion()}) {
      if (region->empty())
        continue;
      for (Operation &inner : region->front().without_terminator()) {
        auto store = dyn_cast<SetOp>(inner);
        if (!store && !speculatable(&inner))
          return success();
        if (store && !op->isAncestor(store.getRef().getParentBlock()->getParentOp())) {
          if (!llvm::is_contained(changed, store.getRef()))
            changed.push_back(store.getRef());
        }
        if (store)
          stores.push_back(store);
      }
    }
    // What each changed object holds before, and at the end of each arm.
    Operation *thenEnd = op.thenYield(), *elseEnd = op.getElseRegion().empty() ? nullptr : op.elseYield();
    SmallVector<Value> before, whenTrue, whenFalse;
    for (Value object : changed) {
      Value start = reaching(object, op), t = reaching(object, thenEnd), f = elseEnd ? reaching(object, elseEnd) : start;
      if (!start || !t || !f || !settled(start) || !settled(t) || !settled(f))
        return success();
      before.push_back(start);
      whenTrue.push_back(t);
      whenFalse.push_back(f);
    }
    // A name rebound in the arms is bound to different objects in them, which would depend on the outcome.
    for (auto [index, result] : llvm::enumerate(op.getResults())) {
      Value t = op.thenYield().getOperand(index), f = elseEnd ? op.elseYield().getOperand(index) : Value();
      if (t == f) {
        result.replaceAllUsesWith(t);
        continue;
      }
      return op.emitError(isa<RefType>(result.getType())
                              ? "a name is bound to different objects in the arms of a conditional on a distribution, so "
                                "which object it means would depend on the outcome"
                              : "a variable that is not a distribution changes in an arm of a conditional on a distribution");
    }
    for (SetOp store : stores)
      if (llvm::is_contained(changed, store.getRef()))
        store.erase();
    for (Region *region : {&op.getThenRegion(), &op.getElseRegion()})
      if (!region->empty())
        for (Operation &inner : llvm::make_early_inc_range(region->front().without_terminator()))
          inner.moveBefore(op);
    OpBuilder builder(op);
    for (auto [index, object] : llvm::enumerate(changed)) {
      if (same(whenTrue[index], before[index]) && same(whenFalse[index], before[index]))
        continue;
      auto select = SelectOp::create(builder, op.getLoc(), TypeRange{whenTrue[index].getType()}, condition.getInput(),
                                     ValueRange{whenTrue[index]}, ValueRange{whenFalse[index]});
      SetOp::create(builder, op.getLoc(), object, select.getResult(0));
    }
    op.erase();
    if (condition->use_empty())
      condition.erase();
    return success();
  }

  void runOnOperation() override {
    SmallVector<scf::IfOp> conditionals;
    getOperation().walk([&](scf::IfOp op) { conditionals.push_back(op); });  // inner ones first
    for (scf::IfOp op : conditionals)
      if (failed(flatten(op)))
        return signalPassFailure();
  }
};

// MLIR has no combinator that repeats passes until nothing changes, and resolving references needs one: folding a condition
// lets an object be promoted, which lets the next condition fold. So this runs the stock passes in a loop until the
// function stops changing.
struct ResolveRefs : PassWrapper<ResolveRefs, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResolveRefs)
  StringRef getArgument() const final { return "resolve-refs"; }
  StringRef getDescription() const final {
    return "Repeat canonicalize, mem2reg and flatten-dist-if until the references are resolved";
  }
  void getDependentDialects(DialectRegistry &registry) const override { registry.insert<DistDialect>(); }

  static std::string text(Operation *op) {
    std::string result;
    llvm::raw_string_ostream stream(result);
    op->print(stream);
    return result;
  }

  void runOnOperation() override {
    OpPassManager round(func::FuncOp::getOperationName());
    if (failed(parsePassPipeline("canonicalize,mem2reg,flatten-dist-if", round)))
      return signalPassFailure();
    std::string previous = text(getOperation());
    for (unsigned i = 0; i < 100; ++i) {
      if (failed(runPipeline(round, getOperation())))
        return signalPassFailure();
      std::string current = text(getOperation());
      if (current == previous)
        return;
      previous = std::move(current);
    }
    getOperation().emitError("the references did not settle");
    signalPassFailure();
  }
};

// Whatever is left of the references after the pipeline has folded what it can is a program the semantics allow but the
// translation cannot decide, or one they refuse.
struct CheckResolved : PassWrapper<CheckResolved, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CheckResolved)
  StringRef getArgument() const final { return "check-resolved"; }
  StringRef getDescription() const final { return "Report the references and conditionals that were not resolved"; }
  void getDependentDialects(DialectRegistry &registry) const override { registry.insert<DistDialect>(); }

  void runOnOperation() override {
    getOperation().walk([&](Operation *op) {
      // `canonicalize` turns a conditional that only picks a reference into a select; if it is still there, the condition
      // is a superposition (or unknown), and the reference it picks would depend on the outcome.
      if (isa<arith::SelectOp, scf::IfOp>(op) && llvm::any_of(op->getResultTypes(), [](Type t) { return isa<RefType>(t); })) {
        op->emitError("a name is bound to different objects in the arms of a conditional on a distribution, so which object "
                      "it means would depend on the outcome");
        return signalPassFailure();
      }
      if (!isa<NewOp, GetOp, SetOp, ShareOp, AssignOp, OverwriteOp, CondOp>(op))
        return;
      Value written = isa<OverwriteOp>(op) ? op->getOperand(0) : isa<AssignOp>(op) ? op->getOperand(0) : Value();
      Value held = written ? reaching(written, op) : Value();
      if (held && kindOf(held) == Kind::Superposed)
        op->emitError("assigning to a name would drop the distribution it holds; only an in-place update changes one");
      else
        op->emitError("could not be resolved: what this name refers to depends on something that is not known when "
                      "compiling");
      signalPassFailure();
    });
  }
};

} // namespace

void GetOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<ForwardGet>(context);
}
void ShareOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<ResolveShare>(context);
}
void AssignOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<ResolveAssign>(context);
}
void OverwriteOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<ResolveOverwrite>(context);
}

void existing_ir::registerFlattenDistIf() {
  PassRegistration<FlattenDistIf>();
  PassRegistration<ResolveRefs>();
}
void existing_ir::registerCheckResolved() { PassRegistration<CheckResolved>(); }

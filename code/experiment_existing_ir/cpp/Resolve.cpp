// Resolving references, kinds and conditionals on a superposition in one analysis and one rewrite.
//
// A variable that may hold a distribution holds a reference to an object, a `memref<!dist.dist<T>>` cell read and
// written with `dist.get` and `dist.set`. What a program means depends on facts that each need the others: which arm of
// a conditional is live, whether a value holds one value (and which), what a cell holds at a point, which cell a
// reference names. They are found together, as in Wegman and Zadeck's conditional constant propagation, by MLIR's
// DataFlowSolver:
//   - `DeadCodeAnalysis` (stock) decides which branches are live, from the constants of their conditions.
//   - `DistConstants` extends the stock `SparseConstantPropagation`. A value is a constant (an IntegerAttr: a distribution
//     of one value is the same fact as the integer), superposed (a UnitAttr) or unknown. A `dist.get` takes its fact from
//     what the cell holds at that point.
//   - `CellRefs` (sparse) says which cell a reference names: an alloca, or a select between references whose condition
//     is a constant, or the result of a conditional.
//   - `CellContents` (dense, forward) says what each cell holds before and after every operation: the value last
//     stored, or, after a conditional on a superposition, a join of the two arms that ties the cell to the condition.
// `resolve-dist` runs the solver once and rewrites once from what it found: reads become the stored values, questions
// (`is_single`, `cond`) and constant values become constants, conditionals with a constant condition keep one arm,
// conditionals on a superposition become both arms and a `dist.select` per tied cell, and the cells go. What is left is
// for stock `canonicalize` (constant selects, dead operations) and for the program to be refused if a fact was not found.
#include "Dialects.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::dataflow;
using namespace existing_ir::dist;

namespace {

// A cell is a rank 0 memref of a distribution; these are its reads and writes.
bool isRead(Operation *op) { auto load = dyn_cast<memref::LoadOp>(op); return load && isa<DistType>(load.getType()); }
bool isWrite(Operation *op) { auto store = dyn_cast<memref::StoreOp>(op); return store && isa<DistType>(store.getValue().getType()); }

// ---- facts about values ------------------------------------------------------------------------------------------------

ConstantValue superposed(MLIRContext *context) {
  return ConstantValue(UnitAttr::get(context), context->getLoadedDialect<DistDialect>());
}
ConstantValue unknown() { return ConstantValue::getUnknownConstant(); }
bool isSuperposed(const ConstantValue &fact) {
  return !fact.isUninitialized() && fact.getConstantValue() && isa<UnitAttr>(fact.getConstantValue());
}
std::optional<APInt> constantOf(const ConstantValue &fact) {
  if (fact.isUninitialized() || !fact.getConstantValue())
    return std::nullopt;
  if (auto attr = dyn_cast<IntegerAttr>(fact.getConstantValue()))
    return attr.getValue();
  if (auto point = dyn_cast<PointAttr>(fact.getConstantValue()))
    return point.getValue().getValue();
  return std::nullopt;
}

// ---- which cell a reference names ----------------------------------------------------------------------------------------

struct CellRef {
  enum Kind { Unreached, Cell, Unknown } kind = Unreached;
  Value cell;
  static CellRef join(const CellRef &a, const CellRef &b) {
    if (a.kind == Unreached)
      return b;
    if (b.kind == Unreached || (a.kind == Cell && b.kind == Cell && a.cell == b.cell))
      return a;
    return {Unknown, {}};
  }
  bool operator==(const CellRef &o) const { return kind == o.kind && cell == o.cell; }
  void print(raw_ostream &os) const { os << (kind == Unreached ? "unreached" : kind == Cell ? "cell" : "unknown"); }
};

class CellRefs : public SparseForwardDataFlowAnalysis<Lattice<CellRef>> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult visitOperation(Operation *op, ArrayRef<const Lattice<CellRef> *> operands,
                               ArrayRef<Lattice<CellRef> *> results) override {
    for (Lattice<CellRef> *result : results) {
      if (!isa<MemRefType>(result->getAnchor().getType()))
        continue;
      if (isa<memref::AllocaOp, memref::AllocOp>(op)) {
        propagateIfChanged(result, result->join({CellRef::Cell, result->getAnchor()}));
      } else if (isa<arith::SelectOp>(op)) {  // a copy or the object itself, once the question is answered
        auto *condition = getOrCreateFor<Lattice<ConstantValue>>(getProgramPointAfter(op), op->getOperand(0));
        if (condition->getValue().isUninitialized())
          continue;
        std::optional<APInt> known = constantOf(condition->getValue());
        for (unsigned side : {1u, 2u})
          if (!known || known->getBoolValue() == (side == 1))
            propagateIfChanged(result, result->join(operands[side]->getValue()));
      } else {
        propagateIfChanged(result, result->join({CellRef::Unknown, {}}));
      }
    }
    return success();
  }
  void setToEntryState(Lattice<CellRef> *lattice) override { propagateIfChanged(lattice, lattice->join({CellRef::Unknown, {}})); }
};

// ---- what the cells hold ---------------------------------------------------------------------------------------------------

// What a cell holds: the value last stored; what a conditional on a superposition left, which is tied to its condition;
// or different things on different paths.
struct Content {
  enum Kind { Store, Merge, Conflict } kind = Conflict;
  Value value;                                  // Store: the value
  Operation *conditional = nullptr;             // Merge: the conditional that tied the cell,
  std::shared_ptr<std::vector<Content>> parts;  // and what the cell held before it, and at the end of its true and false arms
  bool operator==(const Content &o) const {
    return kind == o.kind && value == o.value && conditional == o.conditional && (parts == o.parts || (parts && o.parts && *parts == *o.parts));
  }
};

struct Contents {
  llvm::MapVector<Value, Content> cells;
  bool clobbered = false;  // something was stored through a reference that is not known: nothing can be forwarded
  bool operator==(const Contents &o) const {
    if (clobbered != o.clobbered || cells.size() != o.cells.size())
      return false;
    return llvm::all_of(cells, [&](auto &entry) {
      auto it = o.cells.find(entry.first);
      return it != o.cells.end() && it->second == entry.second;
    });
  }
  void join(const Contents &o) {
    clobbered |= o.clobbered;
    for (auto &[cell, content] : o.cells) {
      auto [it, inserted] = cells.insert({cell, content});
      if (!inserted && !(it->second == content))
        it->second = Content();
    }
  }
};

class Cells : public AbstractDenseLattice {
public:
  using AbstractDenseLattice::AbstractDenseLattice;
  bool known = false;
  Contents contents;

  ChangeResult set(const Contents &next) {
    if (known && contents == next)
      return ChangeResult::NoChange;
    known = true;
    contents = next;
    return ChangeResult::Change;
  }
  ChangeResult join(const AbstractDenseLattice &rhs) override {
    const auto &other = static_cast<const Cells &>(rhs);
    if (!other.known)
      return ChangeResult::NoChange;
    Contents joined = known ? contents : Contents();
    joined.join(other.contents);
    return set(joined);
  }
  void print(raw_ostream &os) const override { os << (known ? "known" : "unknown") << ", " << contents.cells.size() << " cells"; }
};

// What the conditional `ifOp` does: which arms run.
enum class Arms { Pending, Then, Else, Both };

class CellContents : public DenseForwardDataFlowAnalysis<Cells> {
public:
  using DenseForwardDataFlowAnalysis::DenseForwardDataFlowAnalysis;

  // The arms of a conditional, from its condition: a constant takes one; a `dist.cond` of a superposition runs both.
  Arms arms(scf::IfOp op, ProgramPoint *dependent) {
    auto *condition = getOrCreateFor<Lattice<ConstantValue>>(dependent, op.getCondition());
    if (condition->getValue().isUninitialized())
      return Arms::Pending;
    if (std::optional<APInt> known = constantOf(condition->getValue()))
      return known->getBoolValue() ? Arms::Then : Arms::Else;
    return Arms::Both;
  }

  // The state at the end of an arm; before the conditional if the arm is empty.
  const Cells *exit(scf::IfOp op, bool thenArm, ProgramPoint *dependent) {
    Region &region = thenArm ? op.getThenRegion() : op.getElseRegion();
    ProgramPoint *at = region.empty() ? getProgramPointBefore(op) : getProgramPointBefore(region.front().getTerminator());
    return getOrCreateFor<Cells>(dependent, at);
  }

  LogicalResult visitOperation(Operation *op, const Cells &before, Cells *after) override {
    if (!before.known)
      return success();
    Contents next = before.contents;
    if (auto store = dyn_cast<memref::StoreOp>(op); store && isWrite(op)) {
      auto *ref = getOrCreateFor<Lattice<CellRef>>(getProgramPointAfter(op), store.getMemRef());
      if (ref->getValue().kind == CellRef::Unreached)
        return success();
      if (ref->getValue().kind == CellRef::Cell)
        next.cells[ref->getValue().cell] = {Content::Store, store.getValue(), nullptr};
      else
        next.clobbered = true;
    }
    propagateIfChanged(after, after->set(next));
    return success();
  }

  void visitRegionBranchControlFlowTransfer(RegionBranchOpInterface branch, std::optional<unsigned> from,
                                            std::optional<unsigned> to, const Cells &before, Cells *after) override {
    auto op = dyn_cast<scf::IfOp>(branch.getOperation());
    if (!op)
      return join(after, before);
    bool intoArm = from == std::nullopt && to != std::nullopt;
    ProgramPoint *dependent = intoArm ? getProgramPointBefore(&(*to == 0 ? op.getThenRegion() : op.getElseRegion()).front())
                                      : getProgramPointAfter(op);
    Arms taken = arms(op, dependent);
    if (taken == Arms::Pending)
      return;
    // The arm this edge comes from or goes to; the edge from the conditional to itself stands for the missing else.
    bool thenArm = intoArm ? *to == 0 : (from ? *from == 0 : false);
    if ((taken == Arms::Then && !thenArm) || (taken == Arms::Else && thenArm))
      return;
    if (intoArm || taken != Arms::Both || !op.getCondition().getDefiningOp<CondOp>())
      return join(after, before);
    // Leaving a conditional on a superposition: a cell either arm changed is tied to the condition.
    const Cells *onTrue = exit(op, true, dependent), *onFalse = exit(op, false, dependent);
    const Cells *start = getOrCreateFor<Cells>(dependent, getProgramPointBefore(op));
    if (!onTrue->known || !onFalse->known || !start->known)
      return;
    Contents merged = start->contents;
    merged.clobbered |= onTrue->contents.clobbered || onFalse->contents.clobbered;
    for (auto &[cell, content] : start->contents.cells) {
      auto t = onTrue->contents.cells.find(cell), f = onFalse->contents.cells.find(cell);
      if (t != onTrue->contents.cells.end() && f != onFalse->contents.cells.end() &&
          (!(t->second == content) || !(f->second == content)))
        merged.cells[cell] = {Content::Merge, {}, op, std::make_shared<std::vector<Content>>(std::vector<Content>{content, t->second, f->second})};
    }
    Cells result(after->getAnchor());
    (void)result.set(merged);
    join(after, result);
  }

  void setToEntryState(Cells *lattice) override { propagateIfChanged(lattice, lattice->set(Contents())); }
};

// ---- facts about values, extending constant propagation ----------------------------------------------------------------

// Stock constant propagation, which folds the operations (their `fold` hooks) and so knows definite distributions, plus the
// two things folding cannot say: a superposition, and what a read of a cell gives.
class DistConstants : public SparseConstantPropagation {
public:
  using SparseConstantPropagation::SparseConstantPropagation;

  // What a cell holds, as a fact: the stored value's; or for a cell tied to a superposed condition, superposed unless
  // nothing changed. Nullopt while it is not yet known.
  std::optional<ConstantValue> factOf(const Content &content, ProgramPoint *dependent) {
    if (content.kind == Content::Conflict)
      return unknown();
    if (content.kind == Content::Store)
      return getLatticeElementFor(dependent, content.value)->getValue();
    std::optional<ConstantValue> facts[3];
    std::optional<APInt> numbers[3];
    for (int i = 0; i < 3; ++i) {
      if (!(facts[i] = factOf((*content.parts)[i], dependent)))
        return std::nullopt;
      numbers[i] = constantOf(*facts[i]);
    }
    if (numbers[0] && numbers[1] && numbers[2] && *numbers[0] == *numbers[1] && *numbers[1] == *numbers[2])
      return *facts[0];
    return superposed(content.conditional->getContext());
  }
  std::optional<ConstantValue> contentFact(const Contents &contents, Value cell, ProgramPoint *dependent) {
    auto it = contents.cells.find(cell);
    if (contents.clobbered || it == contents.cells.end())
      return unknown();
    return factOf(it->second, dependent);
  }

  LogicalResult visitOperation(Operation *op, ArrayRef<const Lattice<ConstantValue> *> operands,
                               ArrayRef<Lattice<ConstantValue> *> results) override {
    if (!isa<DistDialect>(op->getDialect()) && !isRead(op))
      return SparseConstantPropagation::visitOperation(op, operands, results);
    ProgramPoint *here = getProgramPointAfter(op);
    std::optional<ConstantValue> fact;
    if (auto read = dyn_cast<memref::LoadOp>(op)) {
      auto *ref = getOrCreateFor<Lattice<CellRef>>(here, read.getMemRef());
      auto *state = getOrCreateFor<Cells>(here, getProgramPointBefore(op));
      if (ref->getValue().kind == CellRef::Unreached || !state->known)
        return success();
      fact = ref->getValue().kind == CellRef::Cell ? contentFact(state->contents, ref->getValue().cell, here) : unknown();
    } else if (isa<UniformOp>(op)) {
      fact = superposed(op->getContext());
    } else if (isa<AddIOp, SubIOp, SltOp, AddAssignOp, SubAssignOp, InstanceOp>(op) &&
               llvm::any_of(operands, [](auto *operand) { return isSuperposed(operand->getValue()); })) {
      fact = superposed(op->getContext());  // a result computed from a superposition is one
    } else {
      return SparseConstantPropagation::visitOperation(op, operands, results);  // the folds of the operations
    }
    if (fact)
      propagateIfChanged(results[0], results[0]->join(*fact));
    return success();
  }
};

// ---- the rewrite --------------------------------------------------------------------------------------------------------

struct Resolve : PassWrapper<Resolve, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Resolve)
  StringRef getArgument() const final { return "resolve-dist"; }
  StringRef getDescription() const final {
    return "Resolve references, kinds and conditionals on a superposition with one dataflow analysis and one rewrite";
  }
  void getDependentDialects(DialectRegistry &registry) const override { registry.insert<DistDialect, arith::ArithDialect>(); }

  DataFlowSolver *flow = nullptr;
  bool refused = false;

  // What the solver found, read before anything is changed.
  struct Plan {
    SmallVector<std::pair<scf::IfOp, std::optional<bool>>> conditionals;  // in program order, outermost first, and the arm taken
    SmallVector<std::pair<memref::LoadOp, std::optional<Content>>> reads;  // the content each read gives
  } plan;

  ConstantValue fact(Value value) {
    auto *lattice = flow->lookupState<Lattice<ConstantValue>>(value);
    return lattice ? lattice->getValue() : unknown();
  }
  const Cells *cellsAt(ProgramPoint *point) {
    auto *state = flow->lookupState<Cells>(point);
    return state && state->known ? state : nullptr;
  }

  void refuse(Operation *op, StringRef message) {
    op->emitError(message);
    refused = true;
  }

  void collect(Block &block) {
    for (Operation &op : block) {
      if (auto condition = dyn_cast<scf::IfOp>(op)) {
        conditional(condition);
        continue;
      }
      if (auto check = dyn_cast<cf::AssertOp>(op)) {
        std::optional<APInt> holds = constantOf(fact(check.getArg()));
        if (!holds || !holds->getBoolValue())
          refuse(&op, check.getMsg());
        continue;
      }
      if (auto select = dyn_cast<arith::SelectOp>(op); select && isa<MemRefType>(select.getType()) && !constantOf(fact(select.getCondition()))) {
        refuse(&op, "which object a name means could not be decided: it is bound to different objects in the arms of a "
                    "conditional on a distribution, or depends on something that is not known when compiling");
        continue;
      }
      if (auto read = dyn_cast<memref::LoadOp>(op); read && isRead(&op)) {
        const Cells *state = cellsAt(flow->getProgramPointBefore(&op));
        auto *ref = flow->lookupState<Lattice<CellRef>>(read.getMemRef());
        std::optional<Content> content;
        if (state && ref && ref->getValue().kind == CellRef::Cell && !state->contents.clobbered) {
          auto it = state->contents.cells.find(ref->getValue().cell);
          if (it != state->contents.cells.end() && it->second.kind != Content::Conflict)
            content = it->second;
        }
        if (!content)
          refuse(&op, "could not be resolved: what this name refers to depends on something that is not known when compiling");
        plan.reads.push_back({read, content});
      } else if (isa<IsSingleOp>(op) && !constantOf(fact(op.getResult(0)))) {
        refuse(&op, "could not be resolved: what this name refers to depends on something that is not known when compiling");
      }
    }
  }

  // Two values that are the same thing: one value, one cell, or equal constants.
  bool same(Value a, Value b) {
    if (a == b)
      return true;
    if (isa<MemRefType>(a.getType())) {
      auto *x = flow->lookupState<Lattice<CellRef>>(a), *y = flow->lookupState<Lattice<CellRef>>(b);
      return x && y && x->getValue().kind == CellRef::Cell && x->getValue() == y->getValue();
    }
    std::optional<APInt> x = constantOf(fact(a)), y = constantOf(fact(b));
    return x && y && *x == *y;
  }

  void conditional(scf::IfOp op) {
    std::optional<APInt> known = constantOf(fact(op.getCondition()));
    auto condition = op.getCondition().getDefiningOp<CondOp>();
    plan.conditionals.push_back({op, known ? std::optional<bool>(known->getBoolValue()) : std::nullopt});
    if (known) {
      Region &live = known->getBoolValue() ? op.getThenRegion() : op.getElseRegion();
      if (!live.empty())
        collect(live.front());
      return;
    }
    // A conditional on a superposition does both arms speculatively; they may only read, store to cells and allocate.
    bool flat = condition && isSuperposed(fact(condition.getInput()));
    for (Region *region : {&op.getThenRegion(), &op.getElseRegion()})
      if (!region->empty())
        for (Operation &inner : region->front().without_terminator())
          if (!isWrite(&inner) && !isa<scf::IfOp, cf::AssertOp>(inner))
            if (auto effects = getEffectsRecursively(&inner); !effects || llvm::any_of(*effects, [](auto &e) { return isa<MemoryEffects::Write>(e.getEffect()); }))
              flat = false;
    // A name rebound in the arms is bound to different objects in them, which would depend on the outcome.
    if (flat && !op.getElseRegion().empty())
      for (auto [t, f] : llvm::zip(op.thenYield().getOperands(), op.elseYield().getOperands()))
        flat &= same(t, f);
    if (!flat)
      return refuse(op, isa<MemRefType>(op->getNumResults() ? op->getResult(0).getType() : Type())
                            ? "which object a name means could not be decided: it is bound to different objects in the arms of a "
                              "conditional on a distribution, or depends on something that is not known when compiling"
                            : "could not be resolved: this conditional depends on something that is not known when compiling");
    for (Region *region : {&op.getThenRegion(), &op.getElseRegion()})
      if (!region->empty())
        collect(region->front());
  }

  // The value that holds what a cell holds: the one stored, or the select that ties the cell to the condition.
  DenseMap<std::pair<Operation *, Value>, Value> tied;
  // The integer a content holds when every path gives the same one.
  std::optional<APInt> constantOfContent(const Content &content) {
    if (content.kind == Content::Store)
      return constantOf(fact(content.value));
    if (content.kind == Content::Conflict)
      return std::nullopt;
    std::optional<APInt> a = constantOfContent((*content.parts)[0]), b = constantOfContent((*content.parts)[1]),
                         c = constantOfContent((*content.parts)[2]);
    return a && b && c && *a == *b && *b == *c ? a : std::nullopt;
  }
  DenseMap<Value, Value> forwarded;  // a read that has been replaced, and what it was replaced by
  Value valueOf(const Content &content, Value cell) {
    if (content.kind == Content::Store)  // (a stored value may itself be a read, which is replaced first, as reads come in order)
      return forwarded.lookup_or(content.value, content.value);
    if (constantOfContent(content))  // neither arm changed what it held: not tied to the condition
      return valueOf((*content.parts)[0], cell);
    auto key = std::make_pair(content.conditional, cell);
    if (Value done = tied.lookup(key))
      return done;
    auto op = cast<scf::IfOp>(content.conditional);
    Value onTrue = valueOf((*content.parts)[1], cell), onFalse = valueOf((*content.parts)[2], cell);
    OpBuilder builder(op);
    Value select = SelectOp::create(builder, op.getLoc(), TypeRange{onTrue.getType()}, op.getCondition().getDefiningOp<CondOp>().getInput(),
                                    ValueRange{onTrue}, ValueRange{onFalse}).getResult(0);
    return tied[key] = select;
  }

  void apply() {
    // A conditional on a constant is given that constant and inlined by the stock patterns of scf.if (only those operations are
    // visited, so nothing the plan holds is erased). One on a superposition runs both arms: they are moved out, in order.
    SmallVector<Operation *> constants;
    for (auto [op, taken] : plan.conditionals)
      if (taken) {
        OpBuilder builder(op);
        op->setOperand(0, arith::ConstantOp::create(builder, op.getLoc(), builder.getBoolAttr(*taken)));
        constants.push_back(op);
      }
    RewritePatternSet patterns(&getContext());
    scf::IfOp::getCanonicalizationPatterns(patterns, &getContext());
    (void)applyOpPatternsGreedily(constants, std::move(patterns));
    for (auto [op, taken] : plan.conditionals) {
      if (taken)
        continue;
      for (Region *region : {&op.getThenRegion(), &op.getElseRegion()})
        if (!region->empty())
          for (Operation &inner : llvm::make_early_inc_range(region->front().without_terminator()))
            inner.moveBefore(op);
      if (op->getNumResults())
        op.replaceAllUsesWith(op.thenYield()->getOperands());
    }
    for (auto &[read, content] : plan.reads) {
      if (content) {
        auto ref = flow->lookupState<Lattice<CellRef>>(read.getMemRef());
        Value value = valueOf(*content, ref->getValue().cell);
        forwarded[read.getResult()] = value;
        read.getResult().replaceAllUsesWith(value);
      }
      read.erase();
    }
    for (auto &entry : plan.conditionals)
      if (!entry.second)
        entry.first.erase();
    getOperation().walk([](memref::StoreOp store) {
      if (isWrite(store))
        store.erase();
    });
  }

  void runOnOperation() override {
    DataFlowSolver solver;
    solver.load<DeadCodeAnalysis>();
    solver.load<DistConstants>();
    solver.load<CellRefs>();
    solver.load<CellContents>();
    if (failed(solver.initializeAndRun(getOperation())))
      return signalPassFailure();
    flow = &solver;
    collect(getOperation().getBody().front());
    if (refused)
      return signalPassFailure();
    apply();
  }
};

} // namespace

void existing_ir::registerResolveDist() { PassRegistration<Resolve>(); }

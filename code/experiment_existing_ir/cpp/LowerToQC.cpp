// Lower a program of `func` and `arith` to MQT's QC dialect, one rewrite pattern per operation: the same lowering as
// experiments/existing_ir/rewrites/to_qc.py, written against MLIR itself.
//
// Every integer value becomes a register of qubits (a memref of qubits) as wide as MLIR's integer range analysis says
// the value needs; the IR type is only an upper bound. A constant is loaded with X gates on its set bits. A sum or
// difference is a fresh register: the first operand is copied in with CNOTs, the second into a temporary register, and
// a named ripple-carry adder (or its inverse) adds the temporary into the result. The return measures the result into
// a bit register and releases every register.
#include "Dialects.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using mlir::qc::QubitType;

namespace {

using Widths = DenseMap<Value, unsigned>;
Type qubit(MLIRContext *c) { return QubitType::get(c); }

Value allocate(OpBuilder &b, Location loc, unsigned width) {
  return memref::AllocOp::create(b, loc, MemRefType::get({width}, qubit(b.getContext())));
}

SmallVector<Value> load(OpBuilder &b, Location loc, Value reg) {
  SmallVector<Value> qubits;
  for (int64_t k = 0, n = cast<MemRefType>(reg.getType()).getDimSize(0); k < n; ++k) {
    Value index = arith::ConstantIndexOp::create(b, loc, k);
    qubits.push_back(memref::LoadOp::create(b, loc, reg, index));
  }
  return qubits;
}

// X on the target when every control is set.
void controlledX(OpBuilder &b, Location loc, ValueRange controls, Value target) {
  mlir::qc::CtrlOp::create(b, loc, controls, target, [&](Value t) { mlir::qc::XOp::create(b, loc, t); });
}

std::string adderName(unsigned width) { return "add" + std::to_string(width); }

// The ripple-carry adder of Cuccaro et al. on parameters a = 0..w-1, b = w..2w-1 and carry = 2w: b becomes (a + b)
// modulo 2^w, and a and the carry are unchanged. Two's complement needs nothing more.
void defineAdder(ModuleOp module, unsigned width) {
  OpBuilder b(module.getBodyRegion());
  b.setInsertionPointToEnd(module.getBody());
  Location loc = module.getLoc();
  SmallVector<Type> params(2 * width + 1, qubit(b.getContext()));
  auto fn = func::FuncOp::create(b, loc, adderName(width), b.getFunctionType(params, {}));
  fn.setPrivate();
  fn->setAttr("mqt.unitary", b.getUnitAttr());
  Block *block = fn.addEntryBlock();
  b.setInsertionPointToStart(block);
  auto a = [&](unsigned i) { return block->getArgument(i); };
  auto bb = [&](unsigned i) { return block->getArgument(width + i); };
  auto carry = [&](unsigned i) { return i == 0 ? block->getArgument(2 * width) : a(i - 1); };
  for (unsigned i = 0; i < width; ++i) {  // majority
    controlledX(b, loc, {a(i)}, bb(i));
    controlledX(b, loc, {a(i)}, carry(i));
    controlledX(b, loc, {carry(i), bb(i)}, a(i));
  }
  for (unsigned i = width; i-- > 0;) {  // unmajority and add
    controlledX(b, loc, {carry(i), bb(i)}, a(i));
    controlledX(b, loc, {a(i)}, carry(i));
    controlledX(b, loc, {carry(i)}, bb(i));
  }
  func::ReturnOp::create(b, loc);
}

struct Constant : OpRewritePattern<arith::ConstantOp> {
  Constant(MLIRContext *c, const Widths &w) : OpRewritePattern(c), widths(w) {}
  const Widths &widths;
  LogicalResult matchAndRewrite(arith::ConstantOp op, PatternRewriter &rw) const override {
    auto it = widths.find(op.getResult());
    auto value = dyn_cast<IntegerAttr>(op.getValue());
    if (it == widths.end() || !value || !isa<IntegerType>(op.getType()))
      return failure();
    Value reg = allocate(rw, op.getLoc(), it->second);
    APInt bits = value.getValue().sextOrTrunc(it->second);
    SmallVector<Value> qubits = load(rw, op.getLoc(), reg);
    for (unsigned k = 0; k < qubits.size(); ++k)
      if (bits[k])
        mlir::qc::XOp::create(rw, op.getLoc(), qubits[k]);
    rw.replaceOp(op, reg);
    return success();
  }
};

// The first operand is copied into the result and the second into a temporary: sign-extended if narrower,
// truncated if wider, which is right modulo the result's width.
void copy(OpBuilder &b, Location loc, ArrayRef<Value> source, ArrayRef<Value> target) {
  for (unsigned i = 0; i < target.size(); ++i)
    controlledX(b, loc, {source[std::min<size_t>(i, source.size() - 1)]}, target[i]);
}

// A fresh register of `width` qubits holding lhs + rhs (or lhs - rhs): the first operand is copied in, the second is
// copied into a temporary, and the adder (or its inverse) adds the temporary into the result.
Value sum(OpBuilder &rw, Location loc, Value lhs, Value rhs, unsigned width, bool inverse) {
  Value result = allocate(rw, loc, width), operand = allocate(rw, loc, width), carry = allocate(rw, loc, 1);
  SmallVector<Value> left = load(rw, loc, lhs), right = load(rw, loc, rhs), res = load(rw, loc, result),
                     tmp = load(rw, loc, operand), c = load(rw, loc, carry);
  copy(rw, loc, left, res);
  copy(rw, loc, right, tmp);
  SmallVector<Value> args(tmp);
  args.append(res.begin(), res.end());
  args.push_back(c[0]);
  auto callee = FlatSymbolRefAttr::get(rw.getContext(), adderName(width));
  if (!inverse) {
    mlir::qc::CallOp::create(rw, loc, callee, args);
  } else {
    mlir::qc::InvOp::create(rw, loc, args, [&](ValueRange inner) { mlir::qc::CallOp::create(rw, loc, callee, inner); });
  }
  return result;
}

bool isRegister(Value v) { return isa<MemRefType>(v.getType()); }
unsigned registerWidth(Value v) { return cast<MemRefType>(v.getType()).getDimSize(0); }

template <typename Op, bool Inverse>
struct Sum : OpRewritePattern<Op> {
  Sum(MLIRContext *c, const Widths &w) : OpRewritePattern<Op>(c), widths(w) {}
  const Widths &widths;
  LogicalResult matchAndRewrite(Op op, PatternRewriter &rw) const override {
    auto it = widths.find(op.getResult());
    if (it == widths.end() || !isRegister(op.getLhs()) || !isRegister(op.getRhs()))
      return failure();
    rw.replaceOp(op, sum(rw, op.getLoc(), op.getLhs(), op.getRhs(), it->second, Inverse));
    return success();
  }
};

// lhs < rhs is the sign of lhs - rhs, taken one bit wider than either operand so the difference cannot wrap.
Value lessThan(OpBuilder &rw, Location loc, Value lhs, Value rhs) {
  unsigned width = std::max(registerWidth(lhs), registerWidth(rhs)) + 1;
  Value difference = sum(rw, loc, lhs, rhs, width, true);
  Value result = allocate(rw, loc, 1);
  controlledX(rw, loc, {load(rw, loc, difference).back()}, load(rw, loc, result)[0]);
  return result;
}

struct LessThan : OpRewritePattern<arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::CmpIOp op, PatternRewriter &rw) const override {
    if (op.getPredicate() != arith::CmpIPredicate::slt || !isRegister(op.getLhs()) || !isRegister(op.getRhs()))
      return failure();
    rw.replaceOp(op, lessThan(rw, op.getLoc(), op.getLhs(), op.getRhs()));
    return success();
  }
};

// `target += amount` (or `-=`). When nothing reads the target after the update (checked before any lowering, since lowering
// changes the uses), the adder works in the target's own register, which is as wide as every value of the chain of updates
// it belongs to: the amount is copied into a temporary and added into the target, and the target's register is the result.
// Otherwise the result is a fresh register, as for any sum.
template <typename Op, bool Inverse>
struct Update : OpRewritePattern<Op> {
  Update(MLIRContext *c, const Widths &w, const DenseSet<Operation *> &inPlace)
      : OpRewritePattern<Op>(c), widths(w), inPlace(inPlace) {}
  const Widths &widths;
  const DenseSet<Operation *> &inPlace;
  LogicalResult matchAndRewrite(Op op, PatternRewriter &rw) const override {
    auto it = widths.find(op.getResult());
    if (it == widths.end() || !isRegister(op.getTarget()) || !isRegister(op.getAmount()))
      return failure();
    Location loc = op.getLoc();
    Value target = op.getTarget();
    unsigned width = registerWidth(target);
    if (!inPlace.contains(op) || it->second > width) {
      rw.replaceOp(op, sum(rw, loc, target, op.getAmount(), it->second, Inverse));
      return success();
    }
    Value operand = allocate(rw, loc, width), carry = allocate(rw, loc, 1);
    SmallVector<Value> targetQubits = load(rw, loc, target), amount = load(rw, loc, op.getAmount()),
                       temporary = load(rw, loc, operand), c = load(rw, loc, carry);
    copy(rw, loc, amount, temporary);
    SmallVector<Value> args(temporary);
    args.append(targetQubits.begin(), targetQubits.end());
    args.push_back(c[0]);
    auto callee = FlatSymbolRefAttr::get(rw.getContext(), adderName(width));
    if (!Inverse)
      mlir::qc::CallOp::create(rw, loc, callee, args);
    else
      mlir::qc::InvOp::create(rw, loc, args, [&](ValueRange inner) { mlir::qc::CallOp::create(rw, loc, callee, inner); });
    rw.replaceOp(op, target);
    return success();
  }
};

// A lifted integer is the same register: a distribution and an integer are both qubits, and only the state differs. If
// something else still uses the integer, the register is copied first, since the distribution may be updated in place.
struct Lift : OpRewritePattern<existing_ir::dist::LiftOp> {
  Lift(MLIRContext *c, const Widths &w, const DenseSet<Operation *> &reuse) : OpRewritePattern(c), widths(w), reuse(reuse) {}
  const Widths &widths;
  const DenseSet<Operation *> &reuse;  // the lifts whose integer has no other use, decided before any lowering
  LogicalResult matchAndRewrite(existing_ir::dist::LiftOp op, PatternRewriter &rw) const override {
    Value input = op.getInput();
    auto it = widths.find(op.getResult());
    if (it == widths.end() || !isRegister(input))
      return failure();
    if (reuse.contains(op)) {
      rw.replaceOp(op, input);
      return success();
    }
    Value result = allocate(rw, op.getLoc(), it->second);
    copy(rw, op.getLoc(), load(rw, op.getLoc(), input), load(rw, op.getLoc(), result));
    rw.replaceOp(op, result);
    return success();
  }
};

struct DistLessThan : OpRewritePattern<existing_ir::dist::SltOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(existing_ir::dist::SltOp op, PatternRewriter &rw) const override {
    if (!isRegister(op.getLhs()) || !isRegister(op.getRhs()))
      return failure();
    rw.replaceOp(op, lessThan(rw, op.getLoc(), op.getLhs(), op.getRhs()));
    return success();
  }
};

// condition ? a : b, bit by bit: the result starts as b and takes (a xor b) in the bits where the condition is set.
// Both operands have already been computed, so the selection costs one Toffoli per bit.
Value select(OpBuilder &rw, Location loc, Value condition, Value trueValue, Value falseValue, unsigned width) {
  Value result = allocate(rw, loc, width), delta = allocate(rw, loc, width);
  SmallVector<Value> res = load(rw, loc, result), d = load(rw, loc, delta), a = load(rw, loc, trueValue),
                     b = load(rw, loc, falseValue);
  Value bit = load(rw, loc, condition)[0];
  copy(rw, loc, b, res);
  copy(rw, loc, a, d);
  copy(rw, loc, b, d);
  for (unsigned i = 0; i < res.size(); ++i)
    controlledX(rw, loc, {bit, d[i]}, res[i]);
  return result;
}

struct Select : OpRewritePattern<arith::SelectOp> {
  Select(MLIRContext *c, const Widths &w) : OpRewritePattern(c), widths(w) {}
  const Widths &widths;
  LogicalResult matchAndRewrite(arith::SelectOp op, PatternRewriter &rw) const override {
    auto it = widths.find(op.getResult());
    if (it == widths.end() || !isRegister(op.getCondition()) || !isRegister(op.getTrueValue()) ||
        !isRegister(op.getFalseValue()))
      return failure();
    rw.replaceOp(op, select(rw, op.getLoc(), op.getCondition(), op.getTrueValue(), op.getFalseValue(), it->second));
    return success();
  }
};

struct DistSelect : OpRewritePattern<existing_ir::dist::SelectOp> {
  DistSelect(MLIRContext *c, const Widths &w) : OpRewritePattern(c), widths(w) {}
  const Widths &widths;
  LogicalResult matchAndRewrite(existing_ir::dist::SelectOp op, PatternRewriter &rw) const override {
    if (!isRegister(op.getCondition()) || !llvm::all_of(op.getTrueValues(), isRegister) ||
        !llvm::all_of(op.getFalseValues(), isRegister))
      return failure();
    SmallVector<Value> results;
    for (auto [t, f, r] : llvm::zip(op.getTrueValues(), op.getFalseValues(), op.getResults()))
      results.push_back(select(rw, op.getLoc(), op.getCondition(), t, f, widths.lookup(r)));
    rw.replaceOp(op, results);
    return success();
  }
};

// Reading a distribution measures it, and the return measures the register, so the register passes through.
struct DistMeasure : OpRewritePattern<existing_ir::dist::MeasureOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(existing_ir::dist::MeasureOp op, PatternRewriter &rw) const override {
    if (!isRegister(op.getInput()))
      return failure();
    rw.replaceOp(op, op.getInput());
    return success();
  }
};

// An equal superposition of start .. stop - 1: Hadamards on the low bits when it spans a power of two from zero.
// Other intervals need state preparation, which is a later milestone.
struct Uniform : OpRewritePattern<existing_ir::dist::UniformOp> {
  Uniform(MLIRContext *c, const Widths &w) : OpRewritePattern(c), widths(w) {}
  const Widths &widths;
  LogicalResult matchAndRewrite(existing_ir::dist::UniformOp op, PatternRewriter &rw) const override {
    auto it = widths.find(op.getResult());
    uint32_t span = op.getStop() - op.getStart();
    if (it == widths.end() || op.getStart() != 0 || span == 0 || (span & (span - 1)) != 0)
      return rw.notifyMatchFailure(op, "only 0..2^k is supported");
    Value reg = allocate(rw, op.getLoc(), it->second);
    SmallVector<Value> qubits = load(rw, op.getLoc(), reg);
    for (unsigned k = 0; k < llvm::Log2_32(span); ++k)
      mlir::qc::HOp::create(rw, op.getLoc(), qubits[k]);
    rw.replaceOp(op, reg);
    return success();
  }
};

struct Return : OpRewritePattern<func::ReturnOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(func::ReturnOp op, PatternRewriter &rw) const override {
    auto main = op->getParentOfType<func::FuncOp>();
    if (main.getName() != "main" || op.getNumOperands() != 1 || !isa<MemRefType>(op.getOperand(0).getType()))
      return failure();
    Location loc = op.getLoc();
    SmallVector<Value> registers;
    main.walk([&](memref::AllocOp alloc) { registers.push_back(alloc); });
    SmallVector<Value> qubits = load(rw, loc, op.getOperand(0));
    auto type = mlir::cbit::RegisterType::get(rw.getContext(), qubits.size());
    Value bits = mlir::cbit::AllocOp::create(rw, loc, type, mlir::cbit::InitializationAttr::get(rw.getContext(), mlir::cbit::Initialization::Undefined));
    for (unsigned k = 0; k < qubits.size(); ++k) {
      Value index = arith::ConstantIndexOp::create(rw, loc, k);
      Value bit = mlir::qc::MeasureOp::create(rw, loc, qubits[k]);
      mlir::cbit::StoreOp::create(rw, loc, bit, bits, index);
    }
    for (Value reg : registers)
      memref::DeallocOp::create(rw, loc, reg);
    func::ReturnOp::create(rw, loc, bits);
    main.setFunctionType(rw.getFunctionType({}, {type}));
    main->setAttr("mqt.entry_point", rw.getUnitAttr());
    rw.eraseOp(op);
    return success();
  }
};

struct LowerToQC : PassWrapper<LowerToQC, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToQC)
  StringRef getArgument() const final { return "lower-to-qc"; }
  StringRef getDescription() const final { return "Lower integer arithmetic to QC with one register per value"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::qc::QCDialect, mlir::cbit::CBitDialect, existing_ir::dist::DistDialect, memref::MemRefDialect, arith::ArithDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // MLIR's own range analysis sizes each register; no analysis of ours.
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<dataflow::IntegerRangeAnalysis>();
    if (failed(solver.initializeAndRun(module)))
      return signalPassFailure();
    // 1. The width each value needs, from the range analysis. A distribution's width is its element's: the analysis
    //    follows it through the interface on our operations.
    Widths widths;
    module.walk([&](Operation *op) {
      for (Value result : op->getResults()) {
        Type resultType = result.getType();
        if (auto dist = dyn_cast<existing_ir::dist::DistType>(resultType))
          resultType = dist.getElement();
        auto type = dyn_cast<IntegerType>(resultType);
        auto *lattice = solver.lookupState<dataflow::IntegerValueRangeLattice>(result);
        if (!type || !lattice || lattice->getValue().isUninitialized())
          continue;
        const ConstantIntRanges &range = lattice->getValue().getValue();
        unsigned width = std::max(range.smin().getSignificantBits(), range.smax().getSignificantBits());
        widths[result] = std::min(width, type.getWidth());
      }
    });

    // 2. An update may work in its target's register when no other use of the target comes after it. The values of such
    //    a chain of updates share one register, so it is made as wide as the widest of them and no update needs a wider
    //    one than its target has.
    DenseSet<Operation *> inPlace, reuse;
    DenseMap<Value, Value> parent;  // union-find over the values that share a register
    auto find = [&](Value value) {
      while (parent.contains(value) && parent[value] != value)
        value = parent[value];
      return value;
    };
    module.walk([&](Operation *op) {
      // A lift that reuses its integer's register shares it with the distribution, so they are one register.
      if (auto lift = dyn_cast<existing_ir::dist::LiftOp>(op); lift && lift.getInput().hasOneUse()) {
        reuse.insert(op);
        parent.try_emplace(lift.getInput(), lift.getInput());
        parent.try_emplace(lift.getResult(), lift.getResult());
        parent[find(lift.getResult())] = find(lift.getInput());
      }
      if (!isa<existing_ir::dist::AddAssignOp, existing_ir::dist::SubAssignOp>(op))
        return;
      bool laterUse = llvm::any_of(op->getOperand(0).getUsers(), [&](Operation *user) {
        return user != op && (user->getBlock() != op->getBlock() || !user->isBeforeInBlock(op));
      });
      if (laterUse)
        return;
      inPlace.insert(op);
      parent.try_emplace(op->getOperand(0), op->getOperand(0));
      parent.try_emplace(op->getResult(0), op->getResult(0));
      parent[find(op->getResult(0))] = find(op->getOperand(0));
    });
    DenseMap<Value, unsigned> registerWidths;
    for (auto &member : parent)
      registerWidths[find(member.first)] = std::max(registerWidths[find(member.first)], widths.lookup(member.first));
    for (auto &member : parent)
      widths[member.first] = registerWidths[find(member.first)];

    // 3. The adders needed, at the widths the registers now have.
    std::set<unsigned> adders;
    module.walk([&](Operation *op) {
      if (isa<arith::AddIOp, arith::SubIOp, existing_ir::dist::AddIOp, existing_ir::dist::SubIOp,
              existing_ir::dist::AddAssignOp, existing_ir::dist::SubAssignOp>(op))
        adders.insert(widths.lookup(op->getResult(0)));
      // A comparison subtracts in one bit more than its wider operand.
      if (auto cmp = dyn_cast<arith::CmpIOp>(op))
        adders.insert(std::max(widths.lookup(cmp.getLhs()), widths.lookup(cmp.getRhs())) + 1);
      if (auto slt = dyn_cast<existing_ir::dist::SltOp>(op))
        adders.insert(std::max(widths.lookup(slt.getLhs()), widths.lookup(slt.getRhs())) + 1);
    });
    for (unsigned width : adders)
      defineAdder(module, width);

    RewritePatternSet patterns(&getContext());
    patterns.add<Constant>(&getContext(), widths);
    patterns.add<Sum<arith::AddIOp, false>>(&getContext(), widths);
    patterns.add<Sum<arith::SubIOp, true>>(&getContext(), widths);
    patterns.add<Sum<existing_ir::dist::AddIOp, false>, Sum<existing_ir::dist::SubIOp, true>>(&getContext(), widths);
    patterns.add<LessThan, DistLessThan, DistMeasure>(&getContext());
    patterns.add<Select, DistSelect, Uniform>(&getContext(), widths);
    patterns.add<Lift>(&getContext(), widths, reuse);
    patterns.add<Update<existing_ir::dist::AddAssignOp, false>, Update<existing_ir::dist::SubAssignOp, true>>(
        &getContext(), widths, inPlace);
    patterns.add<Return>(&getContext());
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    config.enableFolding(false);
    config.enableConstantCSE(false);
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    if (failed(applyPatternsGreedily(module, std::move(patterns), config)))
      return signalPassFailure();
  }
};

} // namespace

void existing_ir::registerLowerToQC() { PassRegistration<LowerToQC>(); }

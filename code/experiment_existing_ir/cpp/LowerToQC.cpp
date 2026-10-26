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
struct LessThan : OpRewritePattern<arith::CmpIOp> {
  LessThan(MLIRContext *c, const Widths &w) : OpRewritePattern(c), widths(w) {}
  const Widths &widths;
  LogicalResult matchAndRewrite(arith::CmpIOp op, PatternRewriter &rw) const override {
    if (op.getPredicate() != arith::CmpIPredicate::slt || !isRegister(op.getLhs()) || !isRegister(op.getRhs()))
      return failure();
    Location loc = op.getLoc();
    unsigned width = std::max(registerWidth(op.getLhs()), registerWidth(op.getRhs())) + 1;
    Value difference = sum(rw, loc, op.getLhs(), op.getRhs(), width, true);
    Value result = allocate(rw, loc, 1);
    controlledX(rw, loc, {load(rw, loc, difference).back()}, load(rw, loc, result)[0]);
    rw.replaceOp(op, result);
    return success();
  }
};

// condition ? a : b, bit by bit: the result starts as b and takes (a xor b) in the bits where the condition is set.
// Both operands have already been computed, so the selection costs one Toffoli per bit.
struct Select : OpRewritePattern<arith::SelectOp> {
  Select(MLIRContext *c, const Widths &w) : OpRewritePattern(c), widths(w) {}
  const Widths &widths;
  LogicalResult matchAndRewrite(arith::SelectOp op, PatternRewriter &rw) const override {
    auto it = widths.find(op.getResult());
    if (it == widths.end() || !isRegister(op.getCondition()) || !isRegister(op.getTrueValue()) ||
        !isRegister(op.getFalseValue()))
      return failure();
    Location loc = op.getLoc();
    Value result = allocate(rw, loc, it->second), delta = allocate(rw, loc, it->second);
    SmallVector<Value> res = load(rw, loc, result), d = load(rw, loc, delta), a = load(rw, loc, op.getTrueValue()),
                       b = load(rw, loc, op.getFalseValue());
    Value condition = load(rw, loc, op.getCondition())[0];
    copy(rw, loc, b, res);
    copy(rw, loc, a, d);
    copy(rw, loc, b, d);
    for (unsigned i = 0; i < res.size(); ++i)
      controlledX(rw, loc, {condition, d[i]}, res[i]);
    rw.replaceOp(op, result);
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
    Widths widths;
    std::set<unsigned> adders;
    module.walk([&](Operation *op) {
      for (Value result : op->getResults()) {
        auto type = dyn_cast<IntegerType>(result.getType());
        auto *lattice = solver.lookupState<dataflow::IntegerValueRangeLattice>(result);
        if (!type || !lattice || lattice->getValue().isUninitialized())
          continue;
        const ConstantIntRanges &range = lattice->getValue().getValue();
        unsigned width = std::max(range.smin().getSignificantBits(), range.smax().getSignificantBits());
        widths[result] = std::min(width, type.getWidth());
        if (isa<arith::AddIOp, arith::SubIOp>(op))
          adders.insert(widths[result]);
      }
      // A comparison subtracts in one bit more than its wider operand.
      if (auto cmp = dyn_cast<arith::CmpIOp>(op))
        adders.insert(std::max(widths.lookup(cmp.getLhs()), widths.lookup(cmp.getRhs())) + 1);
    });
    for (unsigned width : adders)
      defineAdder(module, width);

    RewritePatternSet patterns(&getContext());
    patterns.add<Constant>(&getContext(), widths);
    patterns.add<Sum<arith::AddIOp, false>>(&getContext(), widths);
    patterns.add<Sum<arith::SubIOp, true>>(&getContext(), widths);
    patterns.add<LessThan, Select, Uniform>(&getContext(), widths);
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

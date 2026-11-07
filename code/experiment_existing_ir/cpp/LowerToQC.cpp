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
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"
#include <cmath>

using namespace mlir;
using mlir::qc::QubitType;

namespace {

using Widths = DenseMap<Value, unsigned>;
Type qubit(MLIRContext *c) { return QubitType::get(c); }

// Every register is annotated with what it is. As allocated it is scratch (a temporary of one operation); the register that holds the
// result of an operation is then marked as that value's (`replaceWithRegisters`). All start in the zero state.
//   `existing_ir.role`:     "scratch" or "value";
//   `existing_ir.register`: for a value, its register number in the plan (`plan-registers`);
//   `existing_ir.initial`:  the state the register is allocated in.
Value allocate(OpBuilder &b, Location loc, unsigned width) {
  auto alloc = memref::AllocOp::create(b, loc, MemRefType::get({width}, qubit(b.getContext())));
  alloc->setAttr("existing_ir.role", b.getStringAttr("scratch"));
  alloc->setAttr("existing_ir.initial", b.getStringAttr("zero"));
  return alloc;
}

// Replace an operation by the registers that hold its results, marking each register that was allocated for one as that value's.
void replaceWithRegisters(PatternRewriter &rw, Operation *op, ValueRange registers) {
  auto numbers = op->getAttrOfType<DenseI64ArrayAttr>("existing_ir.register");
  for (auto [k, reg] : llvm::enumerate(registers)) {
    auto alloc = reg.getDefiningOp<memref::AllocOp>();
    if (alloc && numbers && numbers.asArrayRef()[k] >= 0 && alloc->getAttrOfType<StringAttr>("existing_ir.role").getValue() == "scratch") {
      alloc->setAttr("existing_ir.role", rw.getStringAttr("value"));
      alloc->setAttr("existing_ir.register", rw.getI64IntegerAttr(numbers.asArrayRef()[k]));
    }
  }
  rw.replaceOp(op, registers);
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
    replaceWithRegisters(rw, op, reg);
    return success();
  }
};

// The first operand is copied into the result and the second into a temporary: sign-extended if narrower,
// truncated if wider, which is right modulo the result's width.
void copy(OpBuilder &b, Location loc, ArrayRef<Value> source, ArrayRef<Value> target) {
  for (unsigned i = 0; i < target.size(); ++i)
    controlledX(b, loc, {source[std::min<size_t>(i, source.size() - 1)]}, target[i]);
}

// An operation's temporaries, and how each is returned to zero so that it can be freed at once: a bitwise copy of an operand is
// copied again (the operand has not changed), and a register computed from the operands is cleared by running the gates that
// computed it backwards (`uncompute`). A register that is freed is dead to this operation and to all that follow.
struct Scratch {
  SmallVector<Value> registers;                         // to free
  SmallVector<std::pair<Value, Value>> copies;          // (source, target): target is a bitwise copy of source

  Value add(Value reg) {
    registers.push_back(reg);
    return reg;
  }
  Value copyOf(OpBuilder &b, Location loc, Value source, unsigned width) {
    Value target = add(allocate(b, loc, width));
    copy(b, loc, load(b, loc, source), load(b, loc, target));
    copies.push_back({source, target});
    return target;
  }
  void free(OpBuilder &b, Location loc) {
    for (auto &[source, target] : llvm::reverse(copies))
      copy(b, loc, load(b, loc, source), load(b, loc, target));
    for (Value reg : registers)
      memref::DeallocOp::create(b, loc, reg);
  }
};

// The gates emitted between two points. Run backwards, they undo what they did: `qc.inv` of the same gates, which MQT's own
// `unroll-modifiers` turns into the inverted gates in reverse order.
struct Span {
  Block *block;
  Operation *before, *last = nullptr;  // the operation before the first of the span (none: the block's start), and the last of it
  explicit Span(OpBuilder &b) : block(b.getInsertionBlock()) {
    auto point = b.getInsertionPoint();
    before = point == block->begin() ? nullptr : &*std::prev(point);
  }
  void close(OpBuilder &b) {
    auto point = b.getInsertionPoint();
    last = point == block->begin() ? nullptr : &*std::prev(point);
  }
};

void uncompute(OpBuilder &b, Location loc, const Span &span) {
  SmallVector<Operation *> gates;
  auto it = span.before ? std::next(span.before->getIterator()) : span.block->begin();
  auto end = span.last ? std::next(span.last->getIterator()) : span.block->begin();
  for (; it != end; ++it)
    if (isa<mlir::qc::QCDialect>(it->getDialect()))
      gates.push_back(&*it);
  SmallVector<Value> qubits;
  for (Operation *gate : gates)
    for (Value operand : gate->getOperands())
      if (isa<QubitType>(operand.getType()) && !llvm::is_contained(qubits, operand))
        qubits.push_back(operand);
  mlir::qc::InvOp::create(b, loc, qubits, [&](ValueRange inner) {
    IRMapping mapping;
    for (auto [outer, local] : llvm::zip(qubits, inner))
      mapping.map(outer, local);
    for (Operation *gate : gates)
      b.clone(*gate, mapping);
  });
}

// The second operand of an operation whose gates control on bits of both: when it is the first operand again (`a * a`,
// `x && x`) a copy is made, since one qubit cannot control a gate twice.
Value distinctFrom(OpBuilder &rw, Location loc, Value lhs, Value rhs, Scratch &scratch) {
  if (lhs != rhs)
    return rhs;
  return scratch.copyOf(rw, loc, rhs, cast<MemRefType>(rhs.getType()).getDimSize(0));
}

// A fresh register of `width` qubits holding lhs + rhs (or lhs - rhs): the first operand is copied in, the second is
// copied into a temporary, and the adder (or its inverse) adds the temporary into the result.
Value sum(OpBuilder &rw, Location loc, Value lhs, Value rhs, unsigned width, bool inverse, Scratch &scratch) {
  Value result = allocate(rw, loc, width), operand = scratch.copyOf(rw, loc, rhs, width), carry = scratch.add(allocate(rw, loc, 1));
  SmallVector<Value> left = load(rw, loc, lhs), res = load(rw, loc, result), tmp = load(rw, loc, operand), c = load(rw, loc, carry);
  copy(rw, loc, left, res);
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
    Scratch scratch;
    Value result = sum(rw, op.getLoc(), op.getLhs(), op.getRhs(), it->second, Inverse, scratch);
    scratch.free(rw, op.getLoc());
    replaceWithRegisters(rw, op, result);
    return success();
  }
};

// lhs < rhs is the sign of lhs - rhs, taken one bit wider than either operand so the difference cannot wrap.
Value lessThan(OpBuilder &rw, Location loc, Value lhs, Value rhs, Scratch &scratch) {
  unsigned width = std::max(registerWidth(lhs), registerWidth(rhs)) + 1;
  Span span(rw);
  size_t copies = scratch.copies.size();
  Value difference = scratch.add(sum(rw, loc, lhs, rhs, width, true, scratch));
  span.close(rw);
  Value result = allocate(rw, loc, 1);
  controlledX(rw, loc, {load(rw, loc, difference).back()}, load(rw, loc, result)[0]);
  uncompute(rw, loc, span);  // the difference has served; run it backwards so that its register is zero again
  scratch.copies.resize(copies);  // (and the operand copy made inside the span is cleared by that too)
  return result;
}

// lhs == rhs: the operands' bitwise difference is all zeros. It is computed in a temporary (the first operand copied in,
// the second added with CNOTs, both sign-extended to the wider), every bit is flipped so that equality is all ones, and one
// multi-controlled X sets the result.
Value equal(OpBuilder &rw, Location loc, Value lhs, Value rhs, Scratch &scratch) {
  unsigned width = std::max(registerWidth(lhs), registerWidth(rhs));
  Value difference = scratch.add(allocate(rw, loc, width)), result = allocate(rw, loc, 1);
  SmallVector<Value> d = load(rw, loc, difference), l = load(rw, loc, lhs), r = load(rw, loc, rhs);
  Span span(rw);
  copy(rw, loc, l, d);
  copy(rw, loc, r, d);
  for (Value bit : d)
    mlir::qc::XOp::create(rw, loc, bit);
  span.close(rw);
  controlledX(rw, loc, d, load(rw, loc, result)[0]);
  uncompute(rw, loc, span);
  return result;
}

// The result register of a comparison, flipped when the comparison is the negation of the one computed.
Value negate(OpBuilder &rw, Location loc, Value flag) {
  mlir::qc::XOp::create(rw, loc, load(rw, loc, flag)[0]);
  return flag;
}

// Every signed comparison is `<` or `==`, with the operands swapped or the result negated.
std::optional<Value> compare(OpBuilder &rw, Location loc, arith::CmpIPredicate predicate, Value lhs, Value rhs, Scratch &scratch) {
  using P = arith::CmpIPredicate;
  switch (predicate) {
  case P::slt: return lessThan(rw, loc, lhs, rhs, scratch);
  case P::sgt: return lessThan(rw, loc, rhs, lhs, scratch);
  case P::sge: return negate(rw, loc, lessThan(rw, loc, lhs, rhs, scratch));
  case P::sle: return negate(rw, loc, lessThan(rw, loc, rhs, lhs, scratch));
  case P::eq: return equal(rw, loc, lhs, rhs, scratch);
  case P::ne: return negate(rw, loc, equal(rw, loc, lhs, rhs, scratch));
  default: return std::nullopt;
  }
}

struct Compare : OpRewritePattern<arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::CmpIOp op, PatternRewriter &rw) const override {
    if (!isRegister(op.getLhs()) || !isRegister(op.getRhs()))
      return failure();
    Scratch scratch;
    std::optional<Value> result = compare(rw, op.getLoc(), op.getPredicate(), op.getLhs(), op.getRhs(), scratch);
    if (!result)
      return rw.notifyMatchFailure(op, "only signed comparisons are supported");
    scratch.free(rw, op.getLoc());
    replaceWithRegisters(rw, op, *result);
    return success();
  }
};

// The product, modulo the result's width, by shift and add: for each bit of the first operand, the second operand shifted
// left by that bit's place is copied into a temporary by Toffolis on that bit, added into the result, and the temporary is
// cleared by the same Toffolis. Operands are sign-extended, which is right modulo the width.
Value product(OpBuilder &rw, Location loc, Value lhs, Value rhs, unsigned width, Scratch &scratch) {
  rhs = distinctFrom(rw, loc, lhs, rhs, scratch);
  Value result = allocate(rw, loc, width), partial = scratch.add(allocate(rw, loc, width)), carry = scratch.add(allocate(rw, loc, 1));
  SmallVector<Value> l = load(rw, loc, lhs), r = load(rw, loc, rhs), p = load(rw, loc, partial), c = load(rw, loc, carry);
  SmallVector<Value> args(p), res = load(rw, loc, result);
  args.append(res.begin(), res.end());
  args.push_back(c[0]);
  auto callee = FlatSymbolRefAttr::get(rw.getContext(), adderName(width));
  for (unsigned i = 0; i < width; ++i) {
    Value bit = l[std::min<size_t>(i, l.size() - 1)];
    auto toffolis = [&] {
      for (unsigned k = i; k < width; ++k)
        controlledX(rw, loc, {bit, r[std::min<size_t>(k - i, r.size() - 1)]}, p[k]);
    };
    toffolis();
    mlir::qc::CallOp::create(rw, loc, callee, args);
    toffolis();
  }
  return result;
}

template <typename Op> struct Product : OpRewritePattern<Op> {
  Product(MLIRContext *c, const Widths &w) : OpRewritePattern<Op>(c), widths(w) {}
  const Widths &widths;
  LogicalResult matchAndRewrite(Op op, PatternRewriter &rw) const override {
    auto it = widths.find(op.getResult());
    if (it == widths.end() || !isRegister(op.getLhs()) || !isRegister(op.getRhs()))
      return failure();
    Scratch scratch;
    Value result = product(rw, op.getLoc(), op.getLhs(), op.getRhs(), it->second, scratch);
    scratch.free(rw, op.getLoc());
    replaceWithRegisters(rw, op, result);
    return success();
  }
};

// Bitwise and, or and xor, bit by bit on operands sign-extended to the result's width. xor is CNOTs; and is a Toffoli per
// bit; or is a xor b xor (a and b).
enum class Bitwise { And, Or, Xor };

template <typename Op, Bitwise Kind> struct BitwiseOp : OpRewritePattern<Op> {
  BitwiseOp(MLIRContext *c, const Widths &w) : OpRewritePattern<Op>(c), widths(w) {}
  const Widths &widths;
  LogicalResult matchAndRewrite(Op op, PatternRewriter &rw) const override {
    auto it = widths.find(op.getResult());
    if (it == widths.end() || !isRegister(op.getLhs()) || !isRegister(op.getRhs()))
      return failure();
    Location loc = op.getLoc();
    Scratch scratch;
    Value result = allocate(rw, loc, it->second), rhs = distinctFrom(rw, loc, op.getLhs(), op.getRhs(), scratch);
    SmallVector<Value> l = load(rw, loc, op.getLhs()), r = load(rw, loc, rhs), res = load(rw, loc, result);
    auto at = [](ArrayRef<Value> bits, unsigned i) { return bits[std::min<size_t>(i, bits.size() - 1)]; };
    for (unsigned i = 0; i < res.size(); ++i) {
      if (Kind == Bitwise::And) {
        controlledX(rw, loc, {at(l, i), at(r, i)}, res[i]);
        continue;
      }
      controlledX(rw, loc, {at(l, i)}, res[i]);
      controlledX(rw, loc, {at(r, i)}, res[i]);
      if (Kind == Bitwise::Or)
        controlledX(rw, loc, {at(l, i), at(r, i)}, res[i]);
    }
    scratch.free(rw, loc);
    replaceWithRegisters(rw, op, result);
    return success();
  }
};

template <typename Op> struct DistCompare : OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op, PatternRewriter &rw) const override {
    if (!isRegister(op.getLhs()) || !isRegister(op.getRhs()))
      return failure();
    auto predicate = std::is_same_v<Op, existing_ir::dist::EqOp> ? arith::CmpIPredicate::eq : arith::CmpIPredicate::slt;
    Scratch scratch;
    Value result = *compare(rw, op.getLoc(), predicate, op.getLhs(), op.getRhs(), scratch);
    scratch.free(rw, op.getLoc());
    replaceWithRegisters(rw, op, result);
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
    Scratch scratch;
    if (!inPlace.contains(op) || it->second > width) {
      Value result = sum(rw, loc, target, op.getAmount(), it->second, Inverse, scratch);
      scratch.free(rw, loc);
      replaceWithRegisters(rw, op, result);
      return success();
    }
    Value operand = scratch.copyOf(rw, loc, op.getAmount(), width), carry = scratch.add(allocate(rw, loc, 1));
    SmallVector<Value> targetQubits = load(rw, loc, target), temporary = load(rw, loc, operand), c = load(rw, loc, carry);
    SmallVector<Value> args(temporary);
    args.append(targetQubits.begin(), targetQubits.end());
    args.push_back(c[0]);
    auto callee = FlatSymbolRefAttr::get(rw.getContext(), adderName(width));
    if (!Inverse)
      mlir::qc::CallOp::create(rw, loc, callee, args);
    else
      mlir::qc::InvOp::create(rw, loc, args, [&](ValueRange inner) { mlir::qc::CallOp::create(rw, loc, callee, inner); });
    scratch.free(rw, loc);
    replaceWithRegisters(rw, op, target);
    return success();
  }
};

// A one-bit flag read as an integer: the bits are copied in with zeros above.
struct ZeroExtend : OpRewritePattern<arith::ExtUIOp> {
  ZeroExtend(MLIRContext *c, const Widths &w) : OpRewritePattern(c), widths(w) {}
  const Widths &widths;
  LogicalResult matchAndRewrite(arith::ExtUIOp op, PatternRewriter &rw) const override {
    auto it = widths.find(op.getResult());
    if (it == widths.end() || !isRegister(op.getIn()))
      return failure();
    Location loc = op.getLoc();
    Value result = allocate(rw, loc, it->second);
    SmallVector<Value> in = load(rw, loc, op.getIn()), out = load(rw, loc, result);
    for (unsigned i = 0; i < std::min(in.size(), out.size()); ++i)
      controlledX(rw, loc, {in[i]}, out[i]);
    replaceWithRegisters(rw, op, result);
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
      replaceWithRegisters(rw, op, input);
      return success();
    }
    Value result = allocate(rw, op.getLoc(), it->second);
    copy(rw, op.getLoc(), load(rw, op.getLoc(), input), load(rw, op.getLoc(), result));
    replaceWithRegisters(rw, op, result);
    return success();
  }
};

// condition ? a : b, bit by bit: the result starts as b and takes (a xor b) in the bits where the condition is set.
// Both operands have already been computed, so the selection costs one Toffoli per bit.
Value select(OpBuilder &rw, Location loc, Value condition, Value trueValue, Value falseValue, unsigned width) {
  Scratch scratch;
  Value result = allocate(rw, loc, width), delta = scratch.add(allocate(rw, loc, width));
  SmallVector<Value> res = load(rw, loc, result), d = load(rw, loc, delta), a = load(rw, loc, trueValue),
                     b = load(rw, loc, falseValue);
  Value bit = load(rw, loc, condition)[0];
  copy(rw, loc, b, res);
  copy(rw, loc, a, d);
  copy(rw, loc, b, d);
  for (unsigned i = 0; i < res.size(); ++i)
    controlledX(rw, loc, {bit, d[i]}, res[i]);
  scratch.copies = {{trueValue, delta}, {falseValue, delta}};  // the difference of the two, cleared by copying them in again
  scratch.free(rw, loc);
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
    replaceWithRegisters(rw, op, select(rw, op.getLoc(), op.getCondition(), op.getTrueValue(), op.getFalseValue(), it->second));
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
    replaceWithRegisters(rw, op, results);
    return success();
  }
};

// Reading a distribution measures it, and the return measures the register, so the register passes through.
struct DistMeasure : OpRewritePattern<existing_ir::dist::MeasureOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(existing_ir::dist::MeasureOp op, PatternRewriter &rw) const override {
    if (!isRegister(op.getInput()))
      return failure();
    replaceWithRegisters(rw, op, op.getInput());
    return success();
  }
};

// An equal superposition of 0 .. n - 1 on the low qubits of a register that starts at zero. It is prepared bit by bit from the
// top. Under a prefix whose subtree lies wholly inside 0 .. n - 1 the next bit is an even coin, which is what a rotation by pi / 2
// from zero makes; under a prefix whose subtree lies wholly outside the amplitude is zero, so whatever is done there is
// harmless. Only the one prefix per level whose subtree straddles n needs another angle, and that rotation is controlled on the
// prefix (a negative control where its bit is zero). A power of two straddles nowhere, so it is Hadamards alone.
void prepareUniform(OpBuilder &rw, Location loc, ArrayRef<Value> qubits, uint64_t n) {
  unsigned k = llvm::Log2_64_Ceil(n);
  if (llvm::isPowerOf2_64(n)) {
    for (unsigned j = 0; j < k; ++j)
      mlir::qc::HOp::create(rw, loc, qubits[j]);
    return;
  }
  for (unsigned j = k; j-- > 0;) {
    mlir::qc::RYOp::create(rw, loc, qubits[j], llvm::numbers::pi / 2);
    uint64_t inside = n & ((uint64_t(2) << j) - 1);  // how much of the straddling subtree (2^(j+1) values) lies below n
    if (inside == 0)
      continue;
    uint64_t upper = inside > (uint64_t(1) << j) ? inside - (uint64_t(1) << j) : 0;  // values in the upper half
    double angle = 2 * std::asin(std::sqrt(double(upper) / double(inside)));
    if (angle == llvm::numbers::pi / 2)
      continue;
    SmallVector<Value> controls, flipped;
    for (unsigned i = j + 1; i < k; ++i) {
      controls.push_back(qubits[i]);
      if (!((n >> i) & 1))
        flipped.push_back(qubits[i]);
    }
    for (Value q : flipped)
      mlir::qc::XOp::create(rw, loc, q);
    double correction = angle - llvm::numbers::pi / 2;
    if (controls.empty())
      mlir::qc::RYOp::create(rw, loc, qubits[j], correction);
    else
      mlir::qc::CtrlOp::create(rw, loc, controls, qubits[j], [&](Value t) { mlir::qc::RYOp::create(rw, loc, t, correction); });
    for (Value q : flipped)
      mlir::qc::XOp::create(rw, loc, q);
  }
}

// An equal superposition of start .. stop - 1: 0 .. stop - start - 1 prepared on the low qubits of the result's register, then
// `start` added in place when it is not zero (an offset register loaded with the constant, and a carry, both cleared afterwards).
struct Uniform : OpRewritePattern<existing_ir::dist::UniformOp> {
  Uniform(MLIRContext *c, const Widths &w) : OpRewritePattern(c), widths(w) {}
  const Widths &widths;
  LogicalResult matchAndRewrite(existing_ir::dist::UniformOp op, PatternRewriter &rw) const override {
    auto it = widths.find(op.getResult());
    if (it == widths.end())
      return failure();
    Location loc = op.getLoc();
    int64_t start = int32_t(op.getStart());
    uint64_t n = uint64_t(int64_t(int32_t(op.getStop())) - start);
    Value result = allocate(rw, loc, it->second);
    SmallVector<Value> qubits = load(rw, loc, result);
    prepareUniform(rw, loc, qubits, n);
    if (start != 0) {
      Scratch scratch;
      Value offset = scratch.add(allocate(rw, loc, it->second)), carry = scratch.add(allocate(rw, loc, 1));
      SmallVector<Value> bits = load(rw, loc, offset);
      APInt pattern = APInt(64, start, true).sextOrTrunc(it->second);
      Span constant(rw);  // the constant, set and later cleared by the same X gates
      for (unsigned k = 0; k < bits.size(); ++k)
        if (pattern[k])
          mlir::qc::XOp::create(rw, loc, bits[k]);
      constant.close(rw);
      SmallVector<Value> args(bits);
      args.append(qubits.begin(), qubits.end());
      args.push_back(::load(rw, loc, carry)[0]);
      mlir::qc::CallOp::create(rw, loc, FlatSymbolRefAttr::get(rw.getContext(), adderName(it->second)), args);
      uncompute(rw, loc, constant);
      scratch.free(rw, loc);
    }
    replaceWithRegisters(rw, op, result);
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
    main.walk([&](memref::AllocOp alloc) {
      if (llvm::none_of(alloc->getUsers(), [](Operation *user) { return isa<memref::DeallocOp>(user); }))
        registers.push_back(alloc);  // freed already: the temporaries of the operations
    });
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
    // The registers are the plan of `plan-registers` (Registers.cpp): made here when no pass has made it yet.
    bool planned = false;
    module.walk([&](Operation *op) { planned |= op->hasAttr("existing_ir.qubits"); });
    if (!planned)
      existing_ir::planRegisters(module);
    Widths widths;
    DenseSet<Operation *> inPlace, reuse;
    module.walk([&](Operation *op) {
      if (auto qubits = op->getAttrOfType<DenseI64ArrayAttr>("existing_ir.qubits"))
        for (auto [result, n] : llvm::zip(op->getResults(), qubits.asArrayRef()))
          if (n > 0)
            widths[result] = n;
      if (op->hasAttr("existing_ir.in_place"))
        inPlace.insert(op);
      if (op->hasAttr("existing_ir.reuse"))
        reuse.insert(op);
    });

    // 3. The adders needed, at the widths the registers now have.
    std::set<unsigned> adders;
    module.walk([&](Operation *op) {
      if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, existing_ir::dist::AddIOp, existing_ir::dist::SubIOp,
              existing_ir::dist::MulIOp, existing_ir::dist::AddAssignOp, existing_ir::dist::SubAssignOp>(op))
        adders.insert(widths.lookup(op->getResult(0)));
      // An order comparison subtracts in one bit more than its wider operand; an equality needs no adder.
      if (auto cmp = dyn_cast<arith::CmpIOp>(op); cmp && cmp.getPredicate() != arith::CmpIPredicate::eq &&
                                                    cmp.getPredicate() != arith::CmpIPredicate::ne)
        adders.insert(std::max(widths.lookup(cmp.getLhs()), widths.lookup(cmp.getRhs())) + 1);
      if (auto uniform = dyn_cast<existing_ir::dist::UniformOp>(op); uniform && int32_t(uniform.getStart()) != 0)
        adders.insert(widths.lookup(uniform.getResult()));
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
    patterns.add<Compare, DistCompare<existing_ir::dist::SltOp>, DistCompare<existing_ir::dist::EqOp>, DistMeasure>(&getContext());
    patterns.add<ZeroExtend>(&getContext(), widths);
    patterns.add<Product<arith::MulIOp>, Product<existing_ir::dist::MulIOp>>(&getContext(), widths);
    patterns.add<BitwiseOp<arith::AndIOp, Bitwise::And>, BitwiseOp<arith::OrIOp, Bitwise::Or>, BitwiseOp<arith::XOrIOp, Bitwise::Xor>,
                 BitwiseOp<existing_ir::dist::AndIOp, Bitwise::And>, BitwiseOp<existing_ir::dist::OrIOp, Bitwise::Or>,
                 BitwiseOp<existing_ir::dist::XOrIOp, Bitwise::Xor>>(&getContext(), widths);
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
    if (failed(checkQubitBudget(module)))
      return signalPassFailure();
    // The target information has been used; the QC program is MQT's, and its parser does not know the dialect.
    module->removeAttr(DLTIDialect::kTargetSystemDescAttrName);
  }

  // The most qubits live at once in `main`, in program order: a register is live from its allocation to its deallocation. It is
  // recorded on `main` as `existing_ir.peak_qubits`, and checked against the system's qubits when the module describes one with
  // MLIR's own target information (`dlti.target_system_spec`, device "quantum", property "qubits").
  LogicalResult checkQubitBudget(ModuleOp module) {
    auto main = module.lookupSymbol<func::FuncOp>("main");
    if (!main)
      return success();
    int64_t live = 0, peak = 0;
    SmallVector<int64_t> sizes;
    auto size = [](Value reg) -> int64_t {
      auto type = dyn_cast<MemRefType>(reg.getType());
      return type && isa<QubitType>(type.getElementType()) ? type.getDimSize(0) : 0;
    };
    main.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto alloc = dyn_cast<memref::AllocOp>(op)) {
        live += size(alloc);
        peak = std::max(peak, live);
        if (size(alloc))
          sizes.push_back(size(alloc));
      } else if (auto dealloc = dyn_cast<memref::DeallocOp>(op)) {
        live -= size(dealloc.getMemref());
      }
    });
    main->setAttr("existing_ir.peak_qubits", IntegerAttr::get(IntegerType::get(&getContext(), 64), peak));
    FailureOr<Attribute> budget = dlti::query(module, ArrayRef<StringRef>{"quantum", "qubits"});
    auto available = succeeded(budget) ? dyn_cast<IntegerAttr>(*budget) : IntegerAttr();
    if (!available || peak <= available.getInt())
      return success();
    llvm::sort(sizes, std::greater<int64_t>());
    InFlightDiagnostic error = module.emitError() << "the program needs " << peak << " qubits at once but the system has "
                                                  << available.getInt() << " (over by " << peak - available.getInt()
                                                  << "); registers, largest first:";
    for (int64_t n : sizes)
      error << " " << n;
    return failure();
  }
};

} // namespace

void existing_ir::registerLowerToQC() { PassRegistration<LowerToQC>(); }

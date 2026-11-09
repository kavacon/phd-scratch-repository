// The distribution dialect: a value that holds an equal or computed superposition of integers, with its own type
// because its meaning differs from an integer's.
//
// Reading a distribution is free and can repeat: the readers compute their results out of place, leaving the register
// as it was, and what that adds is entanglement (a result is correlated with everything it was computed from), which
// the status analysis reports. Changing one is not free: the only way is an in-place update (`add_assign`,
// `sub_assign`), which consumes its target, so every other use of the target must come before the update. Classical
// control flow cannot branch on a distribution (scf.if needs an i1, which a distribution is not), and reading one at
// the return is a measurement.
//
// Operands may be a distribution or a plain integer; at least one must be a distribution, so arithmetic on plain
// integers stays in `arith`. The result is always a distribution, correlated with every distribution operand.
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Dist_Dialect : Dialect {
  let name = "dist";
  let cppNamespace = "::existing_ir::dist";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let hasConstantMaterializer = 1;  // a folded definite distribution is a lift of an arith constant
  let dependentDialects = ["::mlir::arith::ArithDialect"];  // `resolve-dist` writes the constants it finds as arith
}

def Dist_DistType : TypeDef<Dist_Dialect, "Dist", [MemRefElementTypeInterface]> {
  let summary = "a superposition of integers of the element type";
  let mnemonic = "dist";
  let parameters = (ins "::mlir::Type":$element);
  let assemblyFormat = "`<` $element `>`";
}
// A definite distribution holding one value, as an attribute: what a fold of a distribution operation gives when every
// operand is known, and what `materializeConstant` turns back into `dist.lift` of an `arith.constant`.
def Dist_PointAttr : AttrDef<Dist_Dialect, "Point", []> {
  let mnemonic = "point";
  let parameters = (ins "::mlir::IntegerAttr":$value);
  let assemblyFormat = "`<` $value `>`";
}
def Dist_Dist : DialectType<Dist_Dialect, CPred<"::llvm::isa<::existing_ir::dist::DistType>($_self)">, "distribution">;
def Dist_DistOrInt : AnyTypeOf<[Dist_Dist, AnySignlessInteger], "a distribution or an integer">;

// Every operation tells MLIR's integer range analysis the range of its results, so register sizes come from upstream's
// analysis even though a distribution is not an integer type.
class Dist_Op<string mnemonic, list<Trait> traits = []>
    : Op<Dist_Dialect, mnemonic, !listconcat(traits, [DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>])>;

// A function of its operands, so it may be removed when unused and two identical ones may be merged. A range and an
// instance are not: each creates a new independent distribution, so two identical ones must stay two. They are marked
// as allocating, which lets an unused one be removed but stops common-subexpression elimination merging them.
class Dist_PureOp<string mnemonic, list<Trait> traits = []> : Dist_Op<mnemonic, !listconcat(traits, [Pure])>;

// An equal superposition of the integers from start up to, not including, stop.
def Dist_UniformOp : Dist_Op<"uniform"> {
  let arguments = (ins I32Attr:$start, I32Attr:$stop);
  let results = (outs Res<Dist_Dist, "a new independent distribution", [MemAlloc]>:$result);
  let assemblyFormat = "$start `,` $stop attr-dict `:` type($result)";
  let hasVerifier = 1;
}

// An integer taken as a one-point distribution. Physically nothing changes: a definite value and a distribution are both a
// register, and this only says that the value may now be updated in place together with distributions. It creates a new
// object, so two lifts of the same integer stay two (as for a range), and `resolve-dist` turns what was lifted
// without ever being made superposed back into integers.
def Dist_LiftOp : Dist_Op<"lift"> {
  let arguments = (ins AnySignlessInteger:$input);
  let results = (outs Res<Dist_Dist, "a new independent distribution", [MemAlloc]>:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

// A definite distribution written as a constant: what a fold of any distribution operation on known operands becomes. The
// lowering writes it back as `dist.lift` of an `arith.constant`. (Not a `lift`, which has an operand: a constant-like
// operation has none, and without that `canonicalize` refolds a materialized lift for ever.)
def Dist_ConstantOp : Dist_Op<"constant", [Pure, ConstantLike]> {
  let arguments = (ins Dist_PointAttr:$value);
  let results = (outs Dist_Dist:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
  let hasFolder = 1;
}

class Dist_BinaryOp<string mnemonic, list<Trait> traits = []> : Dist_PureOp<mnemonic, traits> {
  let hasFolder = 1;
  let arguments = (ins Dist_DistOrInt:$lhs, Dist_DistOrInt:$rhs);
  let results = (outs Dist_Dist:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasVerifier = 1;
}
// Sum and difference computed out of place: a new distribution, correlated with its operands, which are only used.
def Dist_AddIOp : Dist_BinaryOp<"addi", [Commutative]>;
def Dist_SubIOp : Dist_BinaryOp<"subi">;
// Product, modulo the width of the type.
def Dist_MulIOp : Dist_BinaryOp<"muli", [Commutative]>;
// Signed less-than and equality; the result is a distribution of one-bit integers. Every other comparison is one of these
// with the operands swapped or the result negated.
def Dist_SltOp : Dist_BinaryOp<"slt">;
def Dist_EqOp : Dist_BinaryOp<"eq", [Commutative]>;
// Bitwise operations, used on one-bit integers as the booleans `&&`, `||` and `!` (not is xor with one).
def Dist_AndIOp : Dist_BinaryOp<"andi", [Commutative]>;
def Dist_OrIOp : Dist_BinaryOp<"ori", [Commutative]>;
def Dist_XOrIOp : Dist_BinaryOp<"xori", [Commutative]>;

// The value of a computed expression assigned to a new variable: an independent instance with the law of its input.
// It marks the assignment and says nothing about how the instance is made. `resolve-instances` replays the history of
// the input into fresh distributions, and `tie-instances` keeps the instance tied to its input, for comparison.
def Dist_InstanceOp : Dist_Op<"instance"> {
  let arguments = (ins Dist_Dist:$input);
  let results = (outs Res<Dist_Dist, "a new independent distribution", [MemAlloc]>:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

// `target += amount` and `target -= amount`: the result is the updated target, in the target's register when it fits.
// The target is consumed, so its other uses must come first; the amount is only read.
class Dist_UpdateOp<string mnemonic> : Dist_PureOp<mnemonic> {
  let hasFolder = 1;
  let arguments = (ins Dist_DistOrInt:$target, Dist_DistOrInt:$amount);
  let results = (outs Dist_Dist:$result);
  let assemblyFormat = "$target `,` $amount attr-dict `:` type($target) `,` type($amount) `->` type($result)";
  let hasVerifier = 1;
}
def Dist_AddAssignOp : Dist_UpdateOp<"add_assign">;
def Dist_SubAssignOp : Dist_UpdateOp<"sub_assign">;

// For each pair, the true value where the condition holds and the false value where it does not. One operation takes
// all the pairs so the condition, a distribution, is used once however many variables the conditional changes.
def Dist_SelectOp : Dist_PureOp<"select", [AttrSizedOperandSegments]> {
  let arguments = (ins Dist_Dist:$condition, Variadic<Dist_DistOrInt>:$trueValues, Variadic<Dist_DistOrInt>:$falseValues);
  let results = (outs Variadic<Dist_Dist>:$results);
  let assemblyFormat = "$condition `,` `(` $trueValues `)` `,` `(` $falseValues `)` attr-dict `:` type($condition) `,` "
                       "`(` type($trueValues) `)` `,` `(` type($falseValues) `)` `->` `(` type($results) `)`";
  let hasVerifier = 1;
}

// Reading a distribution measures it: the integer it came out as.
def Dist_MeasureOp : Dist_PureOp<"measure"> {
  let arguments = (ins Dist_Dist:$input);
  let results = (outs AnySignlessInteger:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

// ---- references ------------------------------------------------------------------------------------------------------
class Dist_RawOp<string mnemonic, list<Trait> traits = []> : Op<Dist_Dialect, mnemonic, traits>;

// A variable that may hold a distribution holds a reference to an object: a `memref<!dist.dist<T>>` cell, read and written
// with affine.load and affine.store (which `affine-scalrep` forwards and `mem2reg` leaves alone, so a cell stays in memory
// until its value is known). What `y = x` means is an `arith.select` of a copy and of x on whether x holds one value; that
// question and the condition of a conditional are the operations that are new here; `resolve-dist` answers both.

// Reading and writing the cell of an object are `memref.load` and `memref.store` of the rank 0 cell.

// Whether a distribution holds one value: it is made of lifted integers only. Folded from how the value was made (not from
// its range, which cannot tell a select of equal integers on a superposed condition from an integer).
def Dist_IsSingleOp : Dist_RawOp<"is_single", [Pure]> {
  let arguments = (ins Dist_Dist:$input);
  let results = (outs I1:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasFolder = 1;
}

// `scf.if` branches on this: the condition of a conditional as an integer, when it is definite. `resolve-dist` finds it a
// constant and the `scf.if` keeps one arm; one it finds superposed becomes both arms and a `dist.select` per tied cell.
def Dist_CondOp : Dist_RawOp<"cond", [Pure]> {
  let arguments = (ins Dist_Dist:$input);
  let results = (outs I1:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

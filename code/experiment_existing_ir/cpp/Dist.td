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
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Dist_Dialect : Dialect {
  let name = "dist";
  let cppNamespace = "::existing_ir::dist";
  let useDefaultTypePrinterParser = 1;
  let hasCanonicalizer = 1;  // the conditional on a superposition (Refs.cpp)
  let dependentDialects = ["::mlir::arith::ArithDialect"];  // the patterns of `canonicalize` turn lifted arithmetic into arith
}

def Dist_DistType : TypeDef<Dist_Dialect, "Dist", [MemRefElementTypeInterface]> {
  let summary = "a superposition of integers of the element type";
  let mnemonic = "dist";
  let parameters = (ins "::mlir::Type":$element);
  let assemblyFormat = "`<` $element `>`";
}
def Dist_Dist : DialectType<Dist_Dialect, CPred<"::llvm::isa<::existing_ir::dist::DistType>($_self)">, "distribution">;
// A reference to an object: a cell holding a distribution of the element type. A variable that can hold a distribution
// holds a reference (in a `memref.alloca`, which `mem2reg` promotes), so which object a name means is an ordinary value
// that a branch or a loop can yield. The objects are promoted to values by `mem2reg` too, once no reference to one escapes.
def Dist_RefType : TypeDef<Dist_Dialect, "Ref", [MemRefElementTypeInterface]> {
  let summary = "a reference to an object holding a distribution of the element type";
  let mnemonic = "ref";
  let parameters = (ins "::mlir::Type":$element);
  let assemblyFormat = "`<` $element `>`";
}
def Dist_Ref : DialectType<Dist_Dialect, CPred<"::llvm::isa<::existing_ir::dist::RefType>($_self)">, "reference">;
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
// object, so two lifts of the same integer stay two (as for a range), and `recover-definite` turns what was lifted
// without ever being made superposed back into integers.
def Dist_LiftOp : Dist_Op<"lift"> {
  let arguments = (ins AnySignlessInteger:$input);
  let results = (outs Res<Dist_Dist, "a new independent distribution", [MemAlloc]>:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

class Dist_BinaryOp<string mnemonic> : Dist_PureOp<mnemonic> {
  let arguments = (ins Dist_DistOrInt:$lhs, Dist_DistOrInt:$rhs);
  let results = (outs Dist_Dist:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}
// Sum and difference computed out of place: a new distribution, correlated with its operands, which are only used.
def Dist_AddIOp : Dist_BinaryOp<"addi">;
def Dist_SubIOp : Dist_BinaryOp<"subi">;
// Signed less-than; the result is a distribution of one-bit integers.
def Dist_SltOp : Dist_BinaryOp<"slt">;

// The value of a computed expression assigned to a new variable: an independent instance with the law of its input.
// It marks the assignment and says nothing about how the instance is made. `resolve-instances` replays the history of
// the input into fresh distributions, and `tie-instances` keeps the instance tied to its input, for comparison.
def Dist_InstanceOp : Dist_Op<"instance"> {
  let arguments = (ins Dist_Dist:$input);
  let results = (outs Res<Dist_Dist, "a new independent distribution", [MemAlloc]>:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

// `target += amount` and `target -= amount`: the result is the updated target, in the target's register when it fits.
// The target is consumed, so its other uses must come first; the amount is only read.
class Dist_UpdateOp<string mnemonic> : Dist_PureOp<mnemonic> {
  let arguments = (ins Dist_DistOrInt:$target, Dist_DistOrInt:$amount);
  let results = (outs Dist_Dist:$result);
  let assemblyFormat = "$target `,` $amount attr-dict `:` type($target) `,` type($amount) `->` type($result)";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
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
  let hasCanonicalizer = 1;
}

// Reading a distribution measures it: the integer it came out as.
def Dist_MeasureOp : Dist_PureOp<"measure"> {
  let arguments = (ins Dist_Dist:$input);
  let results = (outs AnySignlessInteger:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

// ---- objects ---------------------------------------------------------------------------------------------------------
// The operations on references. `mem2reg` turns `new`, `get` and `set` into values; the others are markers that a
// canonicalization pattern replaces once the value they depend on is known (Refs.cpp). Their effects name the references
// they touch, which is how the patterns see what may change an object.
class Dist_RawOp<string mnemonic, list<Trait> traits = []> : Op<Dist_Dialect, mnemonic, traits>;

// A new object holding `initial`.
def Dist_NewOp : Dist_RawOp<"new", [DeclareOpInterfaceMethods<PromotableAllocationOpInterface>]> {
  let arguments = (ins Dist_Dist:$initial);
  let results = (outs Res<Dist_Ref, "a new object", [MemAlloc]>:$result);
  let assemblyFormat = "$initial attr-dict `:` type($initial) `->` type($result)";
  let hasVerifier = 1;
}

// Reading an object is free and may repeat.
def Dist_GetOp : Dist_RawOp<"get", [DeclareOpInterfaceMethods<PromotableMemOpInterface>]> {
  let arguments = (ins Arg<Dist_Ref, "the object read", [MemRead]>:$ref);
  let results = (outs Dist_Dist:$result);
  let assemblyFormat = "$ref attr-dict `:` type($ref) `->` type($result)";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

// Storing a value changes what every name for the object reads.
def Dist_SetOp : Dist_RawOp<"set", [DeclareOpInterfaceMethods<PromotableMemOpInterface>]> {
  let arguments = (ins Arg<Dist_Ref, "the object written", [MemWrite]>:$ref, Dist_Dist:$value);
  let assemblyFormat = "$ref `,` $value attr-dict `:` type($ref) `,` type($value)";
  let hasVerifier = 1;
}

// `y = x` for a new name y: a copy of an integer, the same object for a distribution. Which one depends on the value the
// object holds, which is only known after folding, so this stays until then. It becomes a new object holding the integer
// when the object holds a definite value, and `x` itself when it holds a superposition.
def Dist_ShareOp : Dist_RawOp<"share"> {
  let arguments = (ins Arg<Dist_Ref, "the object shared", [MemRead]>:$source);
  let results = (outs Dist_Ref:$result);
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

// `y = x` for a name y that already holds an integer: x's value stored into y's object when x holds an integer, and x
// itself when it holds a distribution. The result is the object y means from now on.
def Dist_AssignOp : Dist_RawOp<"assign"> {
  let arguments = (ins Arg<Dist_Ref, "the object of the name assigned", [MemRead, MemWrite]>:$target,
                       Arg<Dist_Ref, "the object assigned", [MemRead]>:$source);
  let results = (outs Dist_Ref:$result);
  let assemblyFormat = "$target `,` $source attr-dict `:` type($target) `,` type($source) `->` type($result)";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

// `y = e` for a name y that already exists: a store, allowed only while y's object holds an integer. Assigning over a
// distribution would drop it, which the semantics refuse.
def Dist_OverwriteOp : Dist_RawOp<"overwrite"> {
  let arguments = (ins Arg<Dist_Ref, "the object overwritten", [MemRead, MemWrite]>:$ref, Dist_Dist:$value);
  let assemblyFormat = "$ref `,` $value attr-dict `:` type($ref) `,` type($value)";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

// The condition of a conditional as an integer, when it is definite. `scf.if` branches on this. A lift of an integer folds
// to that integer, so the `scf.if` is an ordinary one and folds in turn. One that stays is a conditional on a
// superposition, which `flatten-dist-if` turns into both arms and a select.
// `touched` are the objects the arms may change (the references of every variable in scope). Using them here keeps `mem2reg`
// from promoting them across the conditional while it is open: what they hold afterwards depends on how it resolves, and
// `flatten-dist-if` needs the objects in memory to see which arm changed what.
def Dist_CondOp : Dist_RawOp<"cond", [Pure]> {
  let arguments = (ins Dist_Dist:$input, Variadic<Dist_Ref>:$touched);
  let results = (outs I1:$result);
  let assemblyFormat = "$input (`,` $touched^)? attr-dict `:` type($input) (`,` type($touched)^)? `->` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

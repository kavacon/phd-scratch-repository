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
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Dist_Dialect : Dialect {
  let name = "dist";
  let cppNamespace = "::existing_ir::dist";
  let useDefaultTypePrinterParser = 1;
}

def Dist_DistType : TypeDef<Dist_Dialect, "Dist"> {
  let summary = "a superposition of integers of the element type";
  let mnemonic = "dist";
  let parameters = (ins "::mlir::Type":$element);
  let assemblyFormat = "`<` $element `>`";
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

class Dist_BinaryOp<string mnemonic> : Dist_PureOp<mnemonic> {
  let arguments = (ins Dist_DistOrInt:$lhs, Dist_DistOrInt:$rhs);
  let results = (outs Dist_Dist:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasVerifier = 1;
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
}

// `target += amount` and `target -= amount`: the result is the updated target, in the target's register when it fits.
// The target is consumed, so its other uses must come first; the amount is only read.
class Dist_UpdateOp<string mnemonic> : Dist_PureOp<mnemonic> {
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
}

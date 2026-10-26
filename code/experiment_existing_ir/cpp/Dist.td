// The one operation a distribution needs so far: an equal superposition of the integers from start up to, not
// including, stop. Its result is an ordinary integer, so the standard arith and scf operations apply to it.
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferIntRangeInterface.td"

def Dist_Dialect : Dialect {
  let name = "dist";
  let cppNamespace = "::existing_ir::dist";
}

def Dist_UniformOp : Op<Dist_Dialect, "uniform", [DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>]> {
  let arguments = (ins I32Attr:$start, I32Attr:$stop);
  let results = (outs AnySignlessInteger:$result);
}

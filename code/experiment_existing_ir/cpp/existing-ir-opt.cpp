#include "Dialects.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QTensor/IR/QTensorDialect.h"
#include "mqt/Conversion/QCOToQC/QCOToQC.h"
#include "mqt/Conversion/QCToQCO/QCToQCO.h"
#include "mqt/Conversion/QCToQIR/QIRBase/QCToQIRBase.h"
#include "mqt/Dialect/QC/Transforms/Passes.h"
#include "mqt/Support/Passes.h"

// `mlir-opt` with MQT's dialects and passes (QC to QCO, QIR, its optimisations) and our own pass and dialect.
int main(int argc, char **argv) {
  mlir::registerAllPasses();
  registerMQTCompilerPasses();
  mlir::registerQCToQCO();
  mlir::registerQCOToQC();
  mlir::registerQCToQIRBase();
  mlir::qc::registerQCPasses();
  existing_ir::registerLowerToQC();
  existing_ir::registerDistStatus();
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::func::registerInlinerExtension(registry);
  registry.insert<mlir::qc::QCDialect, mlir::qco::QCODialect, mlir::qtensor::QTensorDialect, mlir::cbit::CBitDialect,
                  mlir::mqt::MQTDialect, existing_ir::dist::DistDialect>();
  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "existing-ir-opt\n", registry));
}

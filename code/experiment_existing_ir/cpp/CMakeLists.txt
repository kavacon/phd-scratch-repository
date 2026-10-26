cmake_minimum_required(VERSION 3.20)
project(existing_ir LANGUAGES C CXX)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(CMAKE_CXX_STANDARD 17)
# MQT, built from source by helper/cpp/setup.sh (build.sh passes the two locations): its headers and the static libraries
# of its MLIR part.
if(NOT MQT_SOURCE OR NOT MQT_BUILD)
  message(FATAL_ERROR "Build with build.sh, which says where MQT is")
endif()
file(GLOB_RECURSE MQT_LIBS ${MQT_BUILD}/mlir/lib/*.a)
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
                    ${MQT_SOURCE}/mlir/include ${MQT_BUILD}/mlir/include)
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_TARGET_DEFINITIONS Dist.td)
mlir_tablegen(DistDialect.h.inc -gen-dialect-decls -dialect=dist)
mlir_tablegen(DistDialect.cpp.inc -gen-dialect-defs -dialect=dist)
mlir_tablegen(DistOps.h.inc -gen-op-decls -dialect=dist)
mlir_tablegen(DistOps.cpp.inc -gen-op-defs -dialect=dist)
add_public_tablegen_target(DistIncGen)

add_llvm_executable(existing-ir-opt existing-ir-opt.cpp Dist.cpp LowerToQC.cpp)
add_dependencies(existing-ir-opt DistIncGen)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(existing-ir-opt PRIVATE ${MQT_LIBS} ${dialect_libs} ${conversion_libs}
  MLIRRegisterAllDialects MLIRRegisterAllPasses MLIRFuncInlinerExtension MLIRFuncTransforms MLIRIR MLIRParser MLIRPass MLIRTransforms MLIROptLib MLIRAnalysis MLIRArithTransforms MLIRFuncDialect)

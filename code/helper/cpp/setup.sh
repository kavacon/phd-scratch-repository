#!/bin/sh
# Install the C++ toolchain (Homebrew's LLVM with MLIR, CMake, Ninja) and build MQT Core v4.1.1's MLIR libraries from
# source into code/build/mqt (MQT needs MLIR 23.1 or newer). Every experiment's C++ links against these. About ten minutes the
# first time. MQT's QDMI device layer does not build with Homebrew's clang and is not needed, so only libraries are built.
set -e
cd "$(dirname "$0")"
CODE=$(cd ../.. && pwd)
. ./env.sh
brew install llvm cmake ninja
[ -d "$MQT_SOURCE" ] || git clone --depth 1 --branch v4.1.1 https://github.com/munich-quantum-toolkit/core.git "$MQT_SOURCE"
cmake -S "$MQT_SOURCE" -B "$MQT_BUILD" -G Ninja $COMPILERS $MLIR -DBUILD_MQT_CORE_TESTS=OFF \
  -DBUILD_MQT_CORE_BINDINGS=OFF -DBUILD_MQT_CORE_QDMI_DDSIM_DEVICE=OFF -DBUILD_MQT_CORE_QDMI_SC_DEVICE=OFF -DENABLE_IPO=OFF
ninja -C "$MQT_BUILD" MLIRQCDialect MLIRCBitDialect MLIRQCODialect MLIRQTensorDialect MLIRMQTDialect \
  MLIRQCToQCO MLIRQCOToQC MLIRQCToQIRBase MLIRCBitToMemRef MLIRQCTransforms MLIRQCOTransforms MLIRSupportMQT

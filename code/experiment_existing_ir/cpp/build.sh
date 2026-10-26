#!/bin/sh
# Build existing-ir-opt, which is mlir-opt with MQT's dialects and passes, our `dist` dialect and the `lower-to-qc` pass.
# Needs helper/cpp/setup.sh to have run first.
set -e
cd "$(dirname "$0")"
. ../../helper/cpp/env.sh
cmake -S . -B build -G Ninja $COMPILERS $MLIR
cmake --build build

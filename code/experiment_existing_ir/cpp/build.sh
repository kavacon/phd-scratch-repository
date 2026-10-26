#!/bin/sh
# Build existing-ir-opt, which is mlir-opt with MQT's dialects and passes, our `dist` dialect and the `lower-to-qc` pass.
# Needs helper/cpp/setup.sh to have run first.
set -e
cd "$(dirname "$0")"
CODE=$(cd ../.. && pwd)
. ../../helper/cpp/env.sh
cmake -S . -B "$BUILD/experiment_existing_ir" -G Ninja $COMPILERS $MLIR -DMQT_SOURCE="$MQT_SOURCE" -DMQT_BUILD="$MQT_BUILD"
cmake --build "$BUILD/experiment_existing_ir"

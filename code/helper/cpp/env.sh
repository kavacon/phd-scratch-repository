# Source this with CODE set to the code/ directory: where C++ builds go, and the compiler flags every C++ build in this
# repository uses (Homebrew's LLVM, with the macOS SDK). Everything generated lives under code/build, which git ignores.
BUILD="$CODE/build"
MQT_SOURCE="$BUILD/mqt/source"
MQT_BUILD="$BUILD/mqt/build"
LLVM=/opt/homebrew/opt/llvm
SDK=$(xcrun --show-sdk-path)
COMPILERS="-DCMAKE_C_COMPILER=$LLVM/bin/clang -DCMAKE_CXX_COMPILER=$LLVM/bin/clang++ -DCMAKE_OSX_SYSROOT=$SDK -DCMAKE_BUILD_TYPE=Release"
MLIR="-DMLIR_DIR=$LLVM/lib/cmake/mlir -DLLVM_DIR=$LLVM/lib/cmake/llvm"

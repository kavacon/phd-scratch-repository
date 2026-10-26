# Source this: the compiler flags every C++ build in this repository uses (Homebrew's LLVM, with the macOS SDK).
LLVM=/opt/homebrew/opt/llvm
SDK=$(xcrun --show-sdk-path)
COMPILERS="-DCMAKE_C_COMPILER=$LLVM/bin/clang -DCMAKE_CXX_COMPILER=$LLVM/bin/clang++ -DCMAKE_OSX_SYSROOT=$SDK -DCMAKE_BUILD_TYPE=Release"
MLIR="-DMLIR_DIR=$LLVM/lib/cmake/mlir -DLLVM_DIR=$LLVM/lib/cmake/llvm"

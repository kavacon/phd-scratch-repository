#!/bin/sh
# Undo the C++ setup: remove MQT's sources and build, every experiment's build directory, and the Homebrew packages
# installed for them (llvm, cmake and ninja, added on 2026-10-26 for this); keep the packages if something else needs them.
set -e
cd "$(dirname "$0")"
rm -rf external ../../experiment_*/cpp/build
brew uninstall --formula llvm cmake ninja

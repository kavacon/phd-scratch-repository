#!/bin/sh
# Undo the C++ setup: remove code/build (MQT's sources and build and every experiment's build), and the Homebrew packages
# installed for them (llvm, cmake and ninja, added on 2026-10-26 for this); keep the packages if something else needs them.
set -e
cd "$(dirname "$0")"
rm -rf "$(cd ../.. && pwd)/build"
brew uninstall --formula llvm cmake ninja

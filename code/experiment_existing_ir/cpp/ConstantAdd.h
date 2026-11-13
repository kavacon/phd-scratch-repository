#pragma once
// Adding a constant in place by an increment network, and what it costs, so that the choice between it and the adder can be made from the
// width and the constant alone (see ConstantAdd.cpp), and the lowering builds the same network (see LowerToQC.cpp).
//
// An addition of 2^k is an increment of the bits from k up: bit j (from the top down) is flipped when bits k .. j - 1 are all set, then
// bit k. A subtraction of 2^k is those gates in the reverse order. A constant is a sum of such steps, either its set bits, or its signed
// digits (7 is 8 - 1, two steps instead of three), whichever is cheaper.
//
// The costs are what an X with m controls costs once Qiskit decomposes it without an ancilla (experiment_existing_ir.constant_add.table,
// which a test compares with these figures): the decomposition is a named assumption, and the figures are a model, not a measurement of the
// circuit (Qiskit uses idle qubits when a circuit has them).
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace existing_ir {
constexpr unsigned MAX_CONTROLS = 9;
constexpr int64_t TWO_QUBIT[] = {0, 1, 6, 14, 36, 84, 136, 192, 264, 344};
constexpr int64_t DEPTH[] = {1, 1, 11, 27, 81, 150, 213, 303, 422, 620};
constexpr int64_t PROHIBITIVE = int64_t(1) << 40;  // a gate with more controls than the table has

struct Step {
  unsigned bit;
  bool subtract;
};
struct Network {
  llvm::SmallVector<Step> steps;
  int64_t twoQubit = 0, depth = 0;
};

inline void addCostOfStep(unsigned width, unsigned bit, Network &network) {
  for (unsigned j = bit + 1; j < width; ++j) {
    unsigned controls = j - bit;
    network.twoQubit += controls > MAX_CONTROLS ? PROHIBITIVE : TWO_QUBIT[controls];
    network.depth += controls > MAX_CONTROLS ? PROHIBITIVE : DEPTH[controls];
  }
  network.twoQubit += TWO_QUBIT[0];
  network.depth += DEPTH[0];
}

// The cheaper of the constant's set bits and its signed digits (fewest two-qubit gates, then least depth, then the set bits).
inline Network bestNetwork(unsigned width, uint64_t constant) {
  uint64_t mask = width >= 64 ? ~uint64_t(0) : (uint64_t(1) << width) - 1;
  constant &= mask;
  Network plain, digits;
  for (unsigned k = 0; k < width; ++k)
    if ((constant >> k) & 1) {
      plain.steps.push_back({k, false});
      addCostOfStep(width, k, plain);
    }
  int64_t rest = int64_t(constant);
  for (unsigned k = 0; rest; ++k, rest >>= 1)
    if (rest & 1) {
      int64_t digit = 2 - (rest & 3);  // +1 or -1, leaving the rest even
      rest -= digit;
      if (k < width) {  // a term at or above the width vanishes
        digits.steps.push_back({k, digit < 0});
        addCostOfStep(width, k, digits);
      }
    }
  if (digits.twoQubit < plain.twoQubit || (digits.twoQubit == plain.twoQubit && digits.depth < plain.depth))
    return digits;
  return plain;
}

// The adder (Cuccaro) costs the width alone: 16 two-qubit gates and 24 layers a bit, and 3 for loading the constant.
inline int64_t adderTwoQubit(unsigned width) { return 16 * int64_t(width); }
inline int64_t adderDepth(unsigned width) { return 24 * int64_t(width) + 3; }
} // namespace existing_ir

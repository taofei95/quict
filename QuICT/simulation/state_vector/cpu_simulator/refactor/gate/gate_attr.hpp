#ifndef QUICT_SIM_BACKEND_GATE_TAG_H
#define QUICT_SIM_BACKEND_GATE_TAG_H

namespace gate {
// diagnal gate
constexpr int DIAGNAL = 1 << 0;
// controlled gate
constexpr int CONTROL = 1 << 1;

enum class SpecialTag { None };
}  // namespace gate

#endif

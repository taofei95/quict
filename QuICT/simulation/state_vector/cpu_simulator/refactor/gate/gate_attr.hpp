#ifndef QUICT_CPU_SIM_BACKEND_GATE_ATTR_H
#define QUICT_CPU_SIM_BACKEND_GATE_ATTR_H

namespace gate {
using AttrT = int;
// Nothing special
constexpr const AttrT NOTHING = 0;
// Diagnal gate
constexpr const AttrT DIAGNAL = 1 << 0;
// Controlled gate
constexpr const AttrT CONTROL = 1 << 1;
}  // namespace gate

#endif

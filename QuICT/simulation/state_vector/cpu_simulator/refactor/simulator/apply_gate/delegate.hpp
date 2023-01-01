#ifndef QUICT_CPU_SIM_BACKEND_SIMULATOR_DELEGATE_H
#define QUICT_CPU_SIM_BACKEND_SIMULATOR_DELEGATE_H

#include "../../gate/gate.hpp"

namespace sim {
template <class DType>
class ApplyGateDelegate {
 public:
  ApplyGateDelegate() = default;
  virtual ~ApplyGateDelegate() = default;
  virtual void ApplyGate(size_t q_num, DType *data,
                         const gate::Gate<DType> &gate) = 0;
};
}  // namespace sim

#endif

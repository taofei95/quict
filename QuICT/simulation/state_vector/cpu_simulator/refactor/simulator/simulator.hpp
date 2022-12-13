#ifndef QUICT_SIM_BACKEND_SIMULATOR_H
#define QUICT_SIM_BACKEND_SIMULATOR_H

#include <algorithm>

#include "../gate/gate.hpp"

namespace simulator {
template <typename DType>
class Simulator {
 public:
  Simulator(size_t q_num, DType data = nullptr) : q_num_(q_num), data_(data) {
    if (data_ == nullptr) {
      size_t len = 1U << q_num_;
      data_ = new DType[len];
      std::fill(data_, data_ + len, 0);
      data_[0] = static_cast<DType>(1);
    }
  }
  virtual ~Simulator() {
    if (data_) {
      delete[] data_;
    }
  }

  virtual void ApplyGate(const gate::Gate<DType> &gate);

 private:
  // Simulator-maintained state vector
  DType data_;
  // Total qubit number
  size_t q_num_;
};
}  // namespace simulator

#endif

#ifndef QUICT_CPU_SIM_BACKEND_IGATE_H
#define QUICT_CPU_SIM_BACKEND_IGATE_H

#include "gate_tag.hpp"

namespace gate {

template <typename Float>
class IGate {
 public:
  IGate(int q_num, GateTag tag) : q_num_(q_num_), tag_(tag) {}
  ~IGate() {
    if (data_) {
      delete[] data_;
    }
  }

 private:
  // qubit number
  int q_num_;
  // gate attributes
  GateTag tag_;
  // gate's raw data
  Float *data_ = nullptr;
};

}  // namespace gate

#endif

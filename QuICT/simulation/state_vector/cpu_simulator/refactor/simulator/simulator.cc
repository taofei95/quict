#include "simulator.hpp"

#include <cstddef>
#include <stdexcept>

namespace simulator {
template <typename DType>
void Simulator<DType>::ApplyGate(const gate::Gate<DType> &gate) {
  int64_t iter_cnt = 1LL << (q_num_ - gate.q_num_);
  if (gate.q_num_ == 1) {
    // normal unitary
    for (int64_t iter = 0; iter < iter_cnt; ++iter) {
      size_t t = gate.GetTarg(0);
      size_t base_ind = iter >> t << (t + 1) | (iter & (1LL - (1LL << t)));
      size_t inds[2] = {base_ind, base_ind | (1LL << t)};
      DType vec[2] = {data_[inds[0]], data_[inds[1]]}, res[2];
      res[0] = gate[0] * vec[0] + gate[1] * vec[1];
      res[1] = gate[2] * vec[0] + gate[3] * vec[1];
      data_[inds[0]] = res[0];
      data_[inds[1]] = res[1];
    }
  } else if (gate.q_num_ == 2) {
    // Debug check that gate is transformed;
    assert(gate.GetTarg(0) < gate.GetTarg(1));

    // normal unitary
    for (int64_t iter = 0; iter < iter_cnt; ++iter) {
      size_t t0 = gate.GetTarg(0), t1 = gate.GetTarg(1);
      size_t base_ind = iter >> t1 << (t1 + 1) | (iter & (1LL - (1LL << t1)));
      base_ind = base_ind >> t0 << (t0 + 1) | (base_ind & (1LL - (1LL << t0)));
      size_t inds[4] = {base_ind, base_ind | (1LL << t0),
                        base_ind | (1LL << t1),
                        base_ind | (1LL << t0) | (1LL << t1)};
      DType vec[4], res[4];
      for (int i = 0; i < 4; ++i) {
        vec[i] = data_[inds[i]];
      }
      for (int i = 0; i < 4; ++i) {
        res[i] = static_cast<DType>(0);
        for (int j = 0; j < 4; ++j) {
          res[i] += gate[i * 4 + j] * vec[j];
        }
      }
      for (int i = 0; i < 4; ++i) {
        data_[inds[i]] = res[i];
      }
    }
  } else {
    throw std::runtime_error("Not implemented for gate >= 3 qubits!");
  }
}
}  // namespace simulator

#ifndef QUICT_SIM_BACKEND_SIMULATOR_H
#define QUICT_SIM_BACKEND_SIMULATOR_H

#include <algorithm>
#include <stdexcept>

#include "../gate/gate.hpp"

namespace simulator {
template <class DType>
class Simulator {
 public:
  Simulator(size_t q_num, DType *data = nullptr) : q_num_(q_num) {
    if (data == nullptr) {
      size_t len = 1ULL << q_num_;
      data_ = new DType[len];
      std::fill(data_, data_ + len, 0);
      data_[0] = DType(1);
    }
  }

  ~Simulator() {
    if (data_) {
      delete[] data_;
    }
  }

  DType *Data() const noexcept { return data_; }

  void ApplyNormalizedGate(const gate::Gate<DType> &gate) {
    size_t gq_num = gate.Qnum();
    int64_t iter_cnt = 1LL << (q_num_ - gq_num);
    if (gq_num == 1) {
      // normal unitary
#pragma omp parallel for
      for (int64_t iter = 0; iter < iter_cnt; ++iter) {
        size_t t = gate.Get1Targ();
        size_t base_ind = iter >> t << (t + 1) | (iter & (1LL - (1LL << t)));
        size_t inds[2] = {base_ind, base_ind | (1LL << t)};
        DType vec[2] = {data_[inds[0]], data_[inds[1]]}, res[2];
        res[0] = gate[0] * vec[0] + gate[1] * vec[1];
        res[1] = gate[2] * vec[0] + gate[3] * vec[1];
        data_[inds[0]] = res[0];
        data_[inds[1]] = res[1];
      }
    } else if (gq_num == 2) {
      // Debug check that gate is transformed;
      assert(gate.GetTarg(0) < gate.GetTarg(1));

      // normal unitary
#pragma omp parallel for
      for (int64_t iter = 0; iter < iter_cnt; ++iter) {
        // size_t t0 = gate.GetTarg(0), t1 = gate.GetTarg(1);
        auto [t0, t1] = gate.Get2Targ();
        size_t base_ind = iter >> t1 << (t1 + 1) | (iter & (1LL - (1LL << t1)));
        base_ind =
            base_ind >> t0 << (t0 + 1) | (base_ind & (1LL - (1LL << t0)));
        size_t inds[4] = {base_ind, base_ind | (1LL << t0),
                          base_ind | (1LL << t1),
                          base_ind | (1LL << t0) | (1LL << t1)};
        DType vec[4], res[4];
        for (int i = 0; i < 4; ++i) {
          vec[i] = data_[inds[i]];
        }
        for (int i = 0; i < 4; ++i) {
          res[i] = DType(0);
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
  };

  void ApplyUnormalizedGate(gate::Gate<DType> &gate) {
    gate.Normalize();
    ApplyNormalizedGate(gate);
  }

 private:
  // Simulator-maintained state vector
  DType *data_;
  // Total qubit number
  size_t q_num_;
};
}  // namespace simulator

#endif

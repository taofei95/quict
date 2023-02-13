#ifndef QUICT_SIM_BACKEND_SIMULATOR_APPLY_GATE_IMPL_NAIVE_H
#define QUICT_SIM_BACKEND_SIMULATOR_APPLY_GATE_IMPL_NAIVE_H

#include <stdexcept>

#include "../delegate.hpp"

namespace sim {
template <class DType>
class NaiveApplyGateDelegate : public ApplyGateDelegate<DType> {
 public:
  BackendTag GetBackendTag() const override { return BackendTag::NAIVE; }

  void ApplyGate(size_t q_num, DType *data,
                 const gate::Gate<DType> &gate) override {
    size_t gq_num = gate.Qnum();
    if (gq_num == 1) {
      Apply1BitGate(q_num, data, gate);
    } else if (gq_num == 2) {
      Apply2BitGate(q_num, data, gate);
    } else {
      throw std::runtime_error("Not implemented for gate >= 3 qubits!");
    }
  }

 private:
  inline void Apply1BitGate(size_t q_num, DType *data,
                            const gate::Gate<DType> &gate) const noexcept {
    int64_t iter_cnt = int64_t(1) << (q_num - 1);
    size_t t = gate.Get1Targ();
    size_t pos = q_num - t - 1LL;
    size_t mask0 = (1LL << pos) - 1LL;
    size_t mask1 = ~mask0;

    // 0 ... t ... q-1
    // [     ][     ] (q-1 len)
    // ->
    // [     ]0[     ] (q len)
    // mask1:
    // [1...1][0...0]
    // mask0:
    // [0...0][1...1]

    // normal unitary
#pragma omp parallel for
    for (size_t iter = 0; iter < iter_cnt; ++iter) {
      size_t base_ind = ((iter & mask1) << 1) | (iter & mask0);
      size_t inds[2] = {base_ind, base_ind | (1LL << pos)};
      DType vec[2], res[2];
      vec[0] = data[inds[0]];
      vec[1] = data[inds[1]];
      res[0] = gate[0] * vec[0] + gate[1] * vec[1];
      res[1] = gate[2] * vec[0] + gate[3] * vec[1];
      data[inds[0]] = res[0];
      data[inds[1]] = res[1];
    }
  }

  inline void Apply2BitGate(size_t q_num, DType *data,
                            const gate::Gate<DType> &gate) const noexcept {
    int64_t iter_cnt = int64_t(1) << (q_num - 2);
    auto [t0, t1] = gate.Get2Targ();

    // sorted index.
    size_t s0 = t0, s1 = t1;
    if (s0 > s1) {
      std::swap(s0, s1);
    }

    size_t spos0 = q_num - s0 - 2LL, pos0 = q_num - t0 - 1LL;
    size_t spos1 = q_num - s1 - 1LL, pos1 = q_num - t1 - 1LL;
    size_t mask0 = (1LL << spos1) - 1LL;
    size_t mask1 = ((1LL << spos0) - 1LL) ^ mask0;
    size_t mask2 = size_t(~0) ^ (mask0 | mask1);

    // 0 ... s0 ... s1 ... q-1
    // [     ][     ][     ] (q-2 len)
    // ->
    // [     ]0[     ]0[     ] (q len)
    // mask0:
    // [0...0][0...0][1...1]
    // mask1:
    // [0...0][1...1][0...0]
    // mask2:
    // [1...1][0...0][0...0]

    // normal unitary
#pragma omp parallel for
    for (size_t iter = 0; iter < iter_cnt; ++iter) {
      size_t base_ind =
          ((iter & mask2) << 2) | ((iter & mask1) << 1) | (iter & mask0);
      size_t inds[4];
      inds[0] = base_ind;
      inds[1] = inds[0] | (1LL << pos1);
      inds[2] = inds[0] | (1LL << pos0);
      inds[3] = inds[1] | (1LL << pos0);
      DType vec[4], res[4];
      for (int i = 0; i < 4; ++i) {
        vec[i] = data[inds[i]];
      }
      for (int i = 0; i < 4; ++i) {
        res[i] = DType(0);
        for (int j = 0; j < 4; ++j) {
          res[i] += gate[i * 4 + j] * vec[j];
        }
      }
      for (int i = 0; i < 4; ++i) {
        data[inds[i]] = res[i];
      }
    }
  }
};
}  // namespace sim

#endif

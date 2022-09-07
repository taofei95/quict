//
// Created by LinHe on 2021-09-24.
//

#ifndef SIM_BACK_CIRCUIT_SIMULATOR_H
#define SIM_BACK_CIRCUIT_SIMULATOR_H

#include <complex>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "gate.h"
#include "matricks_simulator.h"
#include "tiny_simulator.h"
#include "utility.h"

namespace QuICT {
template <typename Precision>
class CircuitSimulator {
 public:
  explicit CircuitSimulator() {
    // Build name
    using namespace std;
    static_assert(is_same_v<Precision, double> || is_same_v<Precision, float>,
                  "MaTricksSimulator only supports double/float precision.");
    name_ = "CircuitSimulator";
    if constexpr (std::is_same_v<Precision, double>) {
      name_ += "[double]";
    } else if (std::is_same_v<Precision, float>) {
      name_ += " [float]";
    }
  }

  const std::string &name() { return name_; }

  inline std::complex<Precision> *run(
      uint64_t qubit_num,
      const std::vector<GateDescription<Precision>> &gate_desc_vec,
      std::vector<int> &measure_res, bool keep_state);

  inline std::vector<int> sample(uint64_t qubit_num);

 protected:
  std::string name_;
  uint64_t qubit_num_;
  Precision *real_ = nullptr, *imag_ = nullptr;
  inline static TinySimulator<Precision> tiny_sim_;
  inline static MaTricksSimulator<Precision> matricks_sim_;
};

template <typename Precision>
std::complex<Precision> *CircuitSimulator<Precision>::run(
    uint64_t qubit_num,
    const std::vector<GateDescription<Precision>> &gate_desc_vec,
    std::vector<int> &measure_res, bool keep_state) {
  this->qubit_num_ = qubit_num;
  if (!keep_state || (real_ == nullptr && imag_ == nullptr)) {
    // Initialize state vector
    uint64_t len = 1LL << qubit_num_;
    real_ = new Precision[len];
    imag_ = new Precision[len];
    std::fill(real_, real_ + len, 0);
    std::fill(imag_, imag_ + len, 0);
    real_[0] = 1.0;
  }
  if (qubit_num_ > 4) {  // Can use matricks simulator
    for (const auto &gate_desc : gate_desc_vec) {
      matricks_sim_.apply_gate(qubit_num_, gate_desc, real_, imag_,
                               measure_res);
    }
  } else {  // Only can use plain simulator
    for (const auto &gate_desc : gate_desc_vec) {
      tiny_sim_.apply_gate(qubit_num_, gate_desc, real_, imag_, measure_res);
    }
  }
  auto res = new std::complex<Precision>[1 << qubit_num_];
  combine_complex(qubit_num_, real_, imag_, res);
  return res;
}

template <typename Precision>
std::vector<int> CircuitSimulator<Precision>::sample(uint64_t qubit_num) {
  qubit_num_ = qubit_num;
  if (real_ == nullptr || imag_ == nullptr) {
    throw std::runtime_error("Sample with invalid amplitude vector!");
  }

  size_t len = 1ULL << qubit_num_;
  Precision *real_cpy = new Precision[len];
  Precision *imag_cpy = new Precision[len];
  std::copy(real_, real_ + len, real_cpy);
  std::copy(imag_, imag_ + len, imag_cpy);

  std::vector<GateDescription<Precision>> gate_desc_vec;
  std::vector<int> measure_res;
  for (uint64_t i = 0; i < qubit_num_; ++i) {
    gate_desc_vec.emplace_back("measure", std::vector<uint64_t>(1, i));
  }
  if (qubit_num_ > 4) {  // Can use matricks simulator
    for (const auto &gate_desc : gate_desc_vec) {
      matricks_sim_.apply_gate(qubit_num_, gate_desc, real_cpy, imag_cpy,
                               measure_res);
    }
  } else {  // Only can use plain simulator
    for (const auto &gate_desc : gate_desc_vec) {
      tiny_sim_.apply_gate(qubit_num_, gate_desc, real_cpy, imag_cpy, measure_res);
    }
  }

  delete[] real_cpy;
  delete[] imag_cpy;

  return measure_res;
}
};  // namespace QuICT

#endif  // SIM_BACK_CIRCUIT_SIMULATOR_H

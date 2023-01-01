#ifndef QUICT_CPU_SIM_BACKEND_SIMULATOR_H
#define QUICT_CPU_SIM_BACKEND_SIMULATOR_H

#include <algorithm>
#include <memory>
#include <stdexcept>

#include "../gate/gate.hpp"
#include "naive_delegate.hpp"
#include "apply_gate_delegate.hpp"

namespace sim {
template <class DType>
class Simulator {
 public:
  Simulator(size_t q_num) : q_num_(q_num) {
    size_t len = 1ULL << q_num_;
    data_ = std::shared_ptr<DType[]>(new DType[len]);
    std::fill(data_.get(), data_.get() + len, 0);
    data_[0] = DType(1);

    // TODO: replace with a factory pattern, which detects platform
    // features at runtime to select proper implementation.
    d_ = std::make_unique<NaiveSimDelegate<DType>>();
  }

  Simulator(size_t q_num, std::shared_ptr<DType[]> data)
      : q_num_(q_num), data_(std::move(data)) {}

  virtual ~Simulator() = default;

  inline void ApplyGate(gate::Gate<DType> &gate) {
    // Use `data_.get()` directly to save 1 call of shared_ptr ctor because we
    // guarantee that lifecycle of `data_` outlives this `ApplyGate` call.
    d_->ApplyGate(q_num_, data_.get(), gate);
  }

  std::shared_ptr<DType[]> GetStateVector() const noexcept { return data_; }

  void SetStateVector(std::shared_ptr<DType[]> data) noexcept {
    data_ = std::move(data);
  }

 private:
  // Total qubit number
  size_t q_num_;
  // Platform dependent implementation delegation
  std::unique_ptr<SimulatorDelegate<DType>> d_;
  // Simulator-maintained state vector
  std::shared_ptr<DType[]> data_;
};
}  // namespace sim

#endif

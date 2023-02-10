#ifndef QUICT_CPU_SIM_BACKEND_SIMULATOR_H
#define QUICT_CPU_SIM_BACKEND_SIMULATOR_H

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>

#include "../gate/gate.hpp"
#include "../utility/debug_msg.hpp"
#include "../utility/feat_detect.hpp"
#include "./backends.hpp"
#include "apply_gate/delegate.hpp"
#include "apply_gate/impl/naive.hpp"
#include "apply_gate/impl/x86_64/sse.hpp"

namespace sim {
template <class DType>
class Simulator {
 private:
  inline void BuildBackend(BackendTag tag) {
    switch (tag) {
      case BackendTag::SSE: {
        DEBUG_MSG("Using SSE optimized simulator");
        d_ = std::make_unique<SseApplyGateDelegate<DType>>();
      }
      default: {
        d_ = std::make_unique<NaiveApplyGateDelegate<DType>>();
      }
    }
  }

 public:
  Simulator(size_t q_num, BackendTag tag) : q_num_(q_num) {
    size_t len = 1ULL << q_num_;
    data_ = std::shared_ptr<DType[]>(new DType[len]);
    std::fill(data_.get(), data_.get() + len, 0);
    data_[0] = DType(1);

    BuildBackend(tag);
  }

  Simulator(size_t q_num, std::shared_ptr<DType[]> data, BackendTag tag)
      : q_num_(q_num), data_(std::move(data)) {
    BuildBackend(tag);
  }

  ~Simulator() = default;

  inline void ApplyGate(gate::Gate<DType> &gate) {
    // Use `data_.get()` directly to save 1 call of shared_ptr ctor because we
    // guarantee that lifecycle of `data_` outlives this `ApplyGate` call.
    d_->ApplyGate(q_num_, data_.get(), gate);
  }

  std::shared_ptr<DType[]> GetStateVector() const noexcept { return data_; }

  void SetStateVector(std::shared_ptr<DType[]> data) noexcept {
    data_ = std::move(data);
  }

  inline std::string Spec() const noexcept {
    std::stringstream ss;
    ss << "QuICT CPU Simulation Backend Specification: " << std::endl;
    ss << BuildSpec();
    ss << RuntimeSpec();
    return ss.str();
  }

 private:
  // Total qubit number
  size_t q_num_;
  // Platform dependent implementation delegation
  std::unique_ptr<ApplyGateDelegate<DType>> d_;
  // Simulator-maintained state vector
  std::shared_ptr<DType[]> data_;

  // x86_64 cpu feature detector
  inline static util::Cpu_x86_64_Detector feat_{};

  inline std::string BuildSpec() const noexcept {
    std::stringstream ss;
    ss << "[Compile]" << std::endl;
    ss << "  SSE:      " << FlagCStr(QUICT_SUPPORT_SSE) << std::endl;
    ss << "  SSE2:     " << FlagCStr(QUICT_SUPPORT_SSE2) << std::endl;
    return ss.str();
  }

  inline std::string RuntimeSpec() const noexcept {
    std::stringstream ss;
    ss << "[Runtime]" << std::endl;
    ss << "  Vendor:   " << feat_.GetVendorString() << std::endl;
    ss << "  SSE:      " << FlagCStr(feat_.HW_SSE) << std::endl;
    ss << "  SSE2:     " << FlagCStr(feat_.HW_SSE2) << std::endl;
    ss << "  AVX:      " << FlagCStr(feat_.OS_AVX && feat_.HW_AVX) << std::endl;
    ss << "  AVX2:     " << FlagCStr(feat_.OS_AVX && feat_.HW_AVX2) << std::endl;
    ss << "  AVX512F:  " << FlagCStr(feat_.OS_AVX512 && feat_.HW_AVX512_F)
       << std::endl;
    ss << std::endl;
    return ss.str();
  }

  inline const char *FlagCStr(bool flag) const noexcept {
    return flag ? "Yes" : "No";
  }
};
}  // namespace sim

#endif

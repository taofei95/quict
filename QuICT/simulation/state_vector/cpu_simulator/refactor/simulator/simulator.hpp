#ifndef QUICT_CPU_SIM_BACKEND_SIMULATOR_H
#define QUICT_CPU_SIM_BACKEND_SIMULATOR_H

#include <algorithm>
#include <complex>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>

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
      case BackendTag::NAIVE: {
        d_ = std::make_unique<NaiveApplyGateDelegate<DType>>();
        return;
      }
      case BackendTag::SSE: {
        // DEBUG_MSG("Using SSE optimized simulator");
        d_ = std::make_unique<SseApplyGateDelegate<DType>>();
        return;
      }
      default: {
        throw std::runtime_error("Not support such backend tag");
      }
    }
  }

  inline void BuildName() {
    std::stringstream ss;
    // Check backend
    switch (d_->GetBackendTag()) {
      case BackendTag::NAIVE: {
        ss << "NaiveSimulator";
        break;
      }
      case BackendTag::SSE: {
        ss << "SseSimulator";
        break;
      }
      case BackendTag::AVX: {
        ss << "AvxSimulator";
        break;
      }
      case BackendTag::AVX512: {
        ss << "Avx512Simulator";
        break;
      }
      default: {
        throw std::runtime_error("Not support such backend tag");
      }
    }

    // Check fp precision
    if constexpr (std::is_same_v<DType, std::complex<float>>) {
      ss << "[f32]";
    } else if constexpr (std::is_same_v<DType, std::complex<double>>) {
      ss << "[f64]";
    }
    name_ = ss.str();
  }

 public:
  Simulator(size_t q_num, BackendTag tag) : q_num_(q_num) {
    size_t len = 1ULL << q_num_;
    data_ = std::shared_ptr<DType[]>(new DType[len]);
    std::fill(data_.get(), data_.get() + len, 0);
    data_[0] = DType(1);

    BuildBackend(tag);
    BuildName();
  }

  Simulator(size_t q_num, std::shared_ptr<DType[]> data, BackendTag tag)
      : q_num_(q_num), data_(std::move(data)) {
    BuildBackend(tag);
    BuildName();
  }

  Simulator(Simulator &&) = default;

  ~Simulator() = default;

  inline void ApplyGate(const gate::Gate<DType> &gate) {
    // Use `data_.get()` directly to save 1 call of shared_ptr ctor because we
    // guarantee that lifecycle of `data_` outlives this `ApplyGate` call.
    d_->ApplyGate(q_num_, data_.get(), gate);
  }

  std::shared_ptr<DType[]> GetStateVector() const noexcept { return data_; }

  void SetStateVector(std::shared_ptr<DType[]> data) noexcept {
    data_ = std::move(data);
  }

  inline std::string PlatformSpec() const noexcept {
    std::stringstream ss;
    ss << "QuICT CPU Simulation Backend Specification: " << std::endl;
    ss << BuildSpec();
    ss << RuntimeSpec();
    return ss.str();
  }

  inline const std::string &GetName() const noexcept { return name_; }

  static inline const util::Cpu_x86_64_Detector &GetHardwareFeature() {
    return hw_feat_;
  }

  inline BackendTag GetBackendTag() const noexcept {
    return d_->GetBackendTag();
  }

 private:
  // Total qubit number
  size_t q_num_;
  // Platform dependent implementation delegation
  std::unique_ptr<ApplyGateDelegate<DType>> d_;
  // Simulator-maintained state vector
  std::shared_ptr<DType[]> data_;
  // Simulator name, including fp precision and backend tag
  std::string name_;

  // x86_64 cpu feature detector
  inline static util::Cpu_x86_64_Detector hw_feat_{};

  inline std::string BuildSpec() const noexcept {
    std::stringstream ss;
#ifdef __SSE__
    constexpr bool QUICT_SUPPORT_SSE = true;
#else
    constexpr bool QUICT_SUPPORT_SSE = false;
#endif

#ifdef __SSE2__
    constexpr bool QUICT_SUPPORT_SSE2 = true;
#else
    constexpr bool QUICT_SUPPORT_SSE2 = false;
#endif

#ifdef __AVX__
    constexpr bool QUICT_SUPPORT_AVX = true;
#else
    constexpr bool QUICT_SUPPORT_AVX = false;
#endif

#ifdef __AVX2__
    constexpr bool QUICT_SUPPORT_AVX2 = true;
#else
    constexpr bool QUICT_SUPPORT_AVX2 = false;
#endif

#ifdef __AVX512F__
    constexpr bool QUICT_SUPPORT_AVX512F = true;
#else
    constexpr bool QUICT_SUPPORT_AVX512F = false;
#endif
    ss << "[Compile]" << std::endl;
    ss << "  SSE:      " << FlagCStr(QUICT_SUPPORT_SSE) << std::endl;
    ss << "  SSE2:     " << FlagCStr(QUICT_SUPPORT_SSE2) << std::endl;
    ss << "  AVX:      " << FlagCStr(QUICT_SUPPORT_AVX) << std::endl;
    ss << "  AVX2:     " << FlagCStr(QUICT_SUPPORT_AVX2) << std::endl;
    ss << "  AVX512F:  " << FlagCStr(QUICT_SUPPORT_AVX512F) << std::endl;
    return ss.str();
  }

  inline std::string RuntimeSpec() const noexcept {
    std::stringstream ss;
    ss << "[Runtime]" << std::endl;
    ss << "  Vendor:   " << hw_feat_.GetVendorString() << std::endl;
    ss << "  SSE:      " << FlagCStr(hw_feat_.HW_SSE) << std::endl;
    ss << "  SSE2:     " << FlagCStr(hw_feat_.HW_SSE2) << std::endl;
    ss << "  AVX:      " << FlagCStr(hw_feat_.OS_AVX && hw_feat_.HW_AVX)
       << std::endl;
    ss << "  AVX2:     " << FlagCStr(hw_feat_.OS_AVX && hw_feat_.HW_AVX2)
       << std::endl;
    ss << "  AVX512F:  " << FlagCStr(hw_feat_.OS_AVX512 && hw_feat_.HW_AVX512_F)
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

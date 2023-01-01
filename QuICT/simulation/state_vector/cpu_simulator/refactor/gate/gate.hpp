#ifndef QUICT_CPU_SIM_BACKEND_GATE_H
#define QUICT_CPU_SIM_BACKEND_GATE_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>

#include "gate_attr.hpp"

namespace gate {

template <typename DType>
class Gate {
 private:
  using DispatchTag = int;
  static constexpr const DispatchTag PRIVATE = 0;
  Gate(DispatchTag dispatch_tag, size_t q_num, DType *data = nullptr,
       AttrT attr = NOTHING)
      : q_num_(q_num), attr_(attr) {
    // if (attr_ & DIAGNAL) {
    //   len_ = 1 << q_num;
    // } else if (attr_ & CONTROL) {
    //   len_ = 1 << (q_num - 1);
    // } else {
    //   // normal unitary
    //   len_ = 2 << q_num;
    // }
    len_ = 1ULL << (q_num_ << 1);
    // data_ = new DType[len_];
    data_ = std::make_unique<DType[]>(len_);
    if (data != nullptr) {
      std::copy(data, data + len_, data_.get());
    } else {
      std::fill(data_.get(), data_.get() + len_, DType(0));
    }
  }

 public:
  Gate() = default;

  ~Gate() = default;

  Gate(Gate &&gate) = default;

  /*
   * @brief Create a single qubit gate. Copy gate data from provided ptr.
   *
   * @param data, gate data.
   * @param attr, gate attributes.
   * */
  Gate(size_t targ0, DType *data = nullptr, AttrT attr = NOTHING)
      : Gate(PRIVATE, 1, data, attr) {
    targ_[0] = targ0;
  }

  /*
   * @brief Create a 2-bit gate. Copy gate data from provided ptr.
   *
   * @param targ0, first target.
   * @param targ2, second target.
   * @param data, gate data.
   * @param attr, gate attributes.
   * */
  Gate(size_t targ0, size_t targ1, DType *data = nullptr, AttrT attr = NOTHING)
      : Gate(PRIVATE, 2, data, attr) {
    targ_[0] = targ0;
    targ_[1] = targ1;
  }

  inline size_t GetTarg(size_t pos) const noexcept {
    // Debug check target access has no out-of-bound error.
    assert(pos < sizeof(targ_) / sizeof(targ_[0]));
    return targ_[pos];
  }

  inline size_t Get1Targ() const noexcept { return GetTarg(0); }

  inline std::pair<size_t, size_t> Get2Targ() const noexcept {
    auto t0 = GetTarg(0), t1 = GetTarg(1);
    return {t0, t1};
  }

  inline constexpr DType &operator[](size_t pos) noexcept {
    // Debug check data access has no out-of-bound error.
    assert(pos < len_);
    return data_[pos];
  }

  inline constexpr const DType &operator[](size_t pos) const noexcept {
    // Debug check data access has no out-of-bound error.
    assert(pos < len_);
    return data_[pos];
  }

  inline size_t Qnum() const noexcept { return q_num_; }

  inline AttrT Attr() const noexcept { return attr_; }

  /*
   * @brief Normalize this gate, ensuring all targs are in ascending order.
   * */
  inline void Normalize() noexcept {
    if (q_num_ == 1) return;
    // Debug check for qubit number.
    assert(q_num_ == 2);

    using std::swap;
    if (targ_[0] > targ_[1]) {
      swap(targ_[0], targ_[1]);
      // swap row-01 and row-10
      for (int j = 0; j < 4; ++j) {
        swap(data_[1 * 4 + j], data_[2 * 4 + j]);
      }
      // swap col-01 and col-10
      for (int i = 0; i < 4; ++i) {
        swap(data_[i * 4 + 1], data_[i * 4 + 2]);
      }
      // TODO: handle special attr
    }
  }

 private:
  // Data length
  size_t len_;
  // Qubit number
  size_t q_num_;
  // Gate attributes
  AttrT attr_;
  // Gate targets (maybe unused)
  std::array<size_t, 2> targ_;
  // Gate's raw data
  std::unique_ptr<DType[]> data_ = nullptr;
};

}  // namespace gate

#endif

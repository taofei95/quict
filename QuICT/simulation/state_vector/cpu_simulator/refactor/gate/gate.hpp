#ifndef QUICT_CPU_SIM_BACKEND_GATE_H
#define QUICT_CPU_SIM_BACKEND_GATE_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>

#include "gate_attr.hpp"

namespace gate {

template <typename DType>
class Gate {
 private:
  using DispatchTag = int;
  static constexpr const DispatchTag PRIVATE = 0;
  Gate(DispatchTag dispatch_tag, size_t q_num, DType *data = nullptr,
       AttrT attr = NOTHING)
      : q_num_(q_num), attr_(attr), data_(data) {
    // if (attr_ & DIAGNAL) {
    //   len_ = 1 << q_num;
    // } else if (attr_ & CONTROL) {
    //   len_ = 1 << (q_num - 1);
    // } else {
    //   // normal unitary
    //   len_ = 2 << q_num;
    // }
    len_ = 2 << q_num_;
    data_ = new DType[len_];
    if (data != nullptr) {
      std::copy(data, data + len_, data_);
    }
  }

 public:
  // Disable default constructor explicitly.
  Gate() = delete;

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

  ~Gate() {
    if (data_) {
      delete[] data_;
    }
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

 private:
  // Data length
  size_t len_;
  // Gate targets (maybe unused)
  size_t targ_[2];
  // Qubit number
  size_t q_num_;
  // Gate attributes
  AttrT attr_;
  // Gate's raw data
  DType *data_;
};

}  // namespace gate

#endif

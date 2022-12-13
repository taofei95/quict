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
 public:
  Gate(size_t q_num, int attr, DType *data = nullptr,
       SpecialTag tag = SpecialTag::None)
      : q_num_(q_num_), attr_(attr), data_(data), tag_(tag) {
    if (attr_ & DIAGNAL) {
      len_ = 1 << q_num;
    } else if (attr_ & CONTROL) {
      len_ = 1 << (q_num - 1);
    } else {
      // normal unitary
      len_ = 2 << q_num;
    }
    if (data_ == nullptr) {
      data_ = new DType[len_];
      memset(data_, sizeof(DType) * len_, 0);
    } else {
      std::copy(data, data + len_, data_);
    }
  }

  Gate(size_t q_num, int attr, size_t targ0, DType *data = nullptr,
       SpecialTag tag = SpecialTag::None)
      : Gate(q_num, attr, data, tag), targ0_(targ0) {}

  Gate(size_t q_num, int attr, size_t targ0, size_t targ1,
       DType *data = nullptr, SpecialTag tag = SpecialTag::None)
      : Gate(q_num, attr, data, tag), targ0_(targ0), targ1_(targ1) {}

  ~Gate() {
    if (data_) {
      delete[] data_;
    }
  }

#ifndef NDEBUG
#define DEBUG_CHECK_ACCESS(pos) assert((pos) < len_)
#else
#define DEBUG_CHECK_ACCESS(pos) (void(pos))
#endif

  constexpr DType &operator[](size_t pos) noexcept {
    DEBUG_CHECK_ACCESS(pos);
    return data_[pos];
  }

  constexpr const DType &operator[](size_t pos) const noexcept {
    DEBUG_CHECK_ACCESS(pos);
    return data_[pos];
  }

 private:
  // Gate's raw data
  DType *data_ = nullptr;
  // Data length
  size_t len_;
  // Gate targets (maybe unused)
  size_t targ0_ = -1, targ1_ = -1;
  // Qubit number
  size_t q_num_;
  // Gate attributes
  int attr_;
  // Tag for special gates. Mostly it's `None`
  SpecialTag tag_;
};

}  // namespace gate

#endif

#ifndef QUICT_SIM_BACKEND_APPLY_GATE_IMPL_X86_64_SSE_H
#define QUICT_SIM_BACKEND_APPLY_GATE_IMPL_X86_64_SSE_H

// On x86_64 architecture, there are 16 XMM registers available.
// So don't worry about register usage. :)
// https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions

#include <immintrin.h>

#include <complex>
#include <stdexcept>
#include <type_traits>

#include "../../delegate.hpp"

namespace sim {
template <class DType>
class SseApplyGateDelegate : public ApplyGateDelegate<DType> {
 public:
  BackendTag GetBackendTag() const override { return BackendTag::SSE; }

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
    if constexpr (std::is_same_v<DType, std::complex<float>>) {
      B1F32(q_num, data, gate);
    } else if constexpr (std::is_same_v<DType, std::complex<double>>) {
      B1F64(q_num, data, gate);
    } else {
      // Always fail.
      static_assert(!sizeof(DType),
                    "Not support types other than float/double.");
    }
  }

  inline void Apply2BitGate(size_t q_num, DType *data,
                            const gate::Gate<DType> &gate) const noexcept {
    if constexpr (std::is_same_v<DType, std::complex<float>>) {
      B2F32(q_num, data, gate);
    } else if constexpr (std::is_same_v<DType, std::complex<double>>) {
      B2F64(q_num, data, gate);
    } else {
      // Always fail.
      static_assert(!sizeof(DType),
                    "Not support types other than float/double.");
    }
  }

  inline void B1F32(size_t q_num, DType *data,
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

    alignas(16) float mptr[8];
    assert((((size_t)mptr) & 0b1111) == 0);  // Check 16 bytes alignment;
    mptr[0] = gate[0].real();
    mptr[1] = gate[1].real();
    mptr[2] = gate[2].real();
    mptr[3] = gate[3].real();
    mptr[4] = gate[0].imag();
    mptr[5] = gate[1].imag();
    mptr[6] = gate[2].imag();
    mptr[7] = gate[3].imag();

    // [mr00, mr01, mr10, mr11]
    __m128 mr = _mm_load_ps(mptr);
    // [mi00, mi01, mi10, mi11]
    __m128 mi = _mm_load_ps(mptr + 4);
#pragma omp parallel for nowait
    for (size_t iter = 0; iter < iter_cnt; ++iter) {
      size_t base_ind = ((iter & mask1) << 1) | (iter & mask0);
      size_t inds[2] = {base_ind, base_ind | (1LL << pos)};
      // Mat-vec complex mul (2x2, 2)
      alignas(16) float tmp[8];
      assert((((size_t)tmp) & 0b1111) == 0);  // Check 16 bytes alignment;
      alignas(16) float vptr[8];
      assert((((size_t)vptr) & 0b1111) == 0);  // Check 16 bytes alignment;
      vptr[0] = data[inds[0]].real();
      vptr[1] = data[inds[1]].real();
      vptr[2] = data[inds[0]].real();
      vptr[3] = data[inds[1]].real();
      vptr[4] = data[inds[0]].imag();
      vptr[5] = data[inds[1]].imag();
      vptr[6] = data[inds[0]].imag();
      vptr[7] = data[inds[1]].imag();

      __m128 vr, vi;
      __m128 tmpv0, tmpv1, tmpv2, tmpv3, tmpv5, tmpv6;

      // [vr0, vr1, vr0, vr1]
      vr = _mm_load_ps(vptr);
      // [vi0, vi1, vi0, vi1]
      vi = _mm_load_ps(vptr + 4);

      tmpv0 = _mm_mul_ps(mr, vr);
      tmpv1 = _mm_mul_ps(mi, vi);
      tmpv2 = _mm_mul_ps(mr, vi);
      tmpv3 = _mm_mul_ps(mi, vr);

      tmpv5 = _mm_sub_ps(tmpv0, tmpv1);  // real part
      tmpv6 = _mm_add_ps(tmpv2, tmpv3);  // imag part

      _mm_store_ps(tmp, tmpv5);
      _mm_store_ps(tmp + 4, tmpv6);

      data[inds[0]].real(tmp[0] + tmp[1]);
      data[inds[1]].real(tmp[2] + tmp[3]);
      data[inds[0]].imag(tmp[4] + tmp[5]);
      data[inds[1]].imag(tmp[6] + tmp[7]);
    }
  }

  inline void B1F64(size_t q_num, DType *data,
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

    alignas(16) double mptr[8];
    assert((((size_t)mptr) & 0b1111) == 0);  // Check 16 bytes alignment;
    mptr[0] = gate[0].real();
    mptr[1] = gate[1].real();
    mptr[2] = gate[2].real();
    mptr[3] = gate[3].real();
    mptr[4] = gate[0].imag();
    mptr[5] = gate[1].imag();
    mptr[6] = gate[2].imag();
    mptr[7] = gate[3].imag();

    // [mr00, mr01]
    __m128d mr0 = _mm_load_pd(mptr);
    // [mr10, mr11]
    __m128d mr1 = _mm_load_pd(mptr + 2);
    // [mi00, mi01]
    __m128d mi0 = _mm_load_pd(mptr + 4);
    // [mi10, mi11]
    __m128d mi1 = _mm_load_pd(mptr + 6);
#pragma omp parallel for nowait
    for (size_t iter = 0; iter < iter_cnt; ++iter) {
      size_t base_ind = ((iter & mask1) << 1) | (iter & mask0);
      size_t inds[2] = {base_ind, base_ind | (1LL << pos)};
      // Mat-vec complex mul (2x2, 2):
      alignas(16) double tmp[4];
      assert((((size_t)tmp) & 0b1111) == 0);  // Check 16 bytes alignment;
      alignas(16) double vptr[4];
      assert((((size_t)vptr) & 0b1111) == 0);  // Check 16 bytes alignment;
      vptr[0] = data[inds[0]].real();
      vptr[1] = data[inds[1]].real();
      vptr[2] = data[inds[0]].imag();
      vptr[3] = data[inds[1]].imag();

      // [vr0, vr1]
      __m128d vr = _mm_load_pd(vptr);
      // [vi0, vi1]
      __m128d vi = _mm_load_pd(vptr + 2);
      __m128d tmpv0, tmpv1, tmpv2, tmpv3;

      //
      // first row
      //
      // real part
      tmpv0 = _mm_mul_pd(mr0, vr);
      tmpv1 = _mm_mul_pd(mi0, vi);
      // imag part
      tmpv2 = _mm_mul_pd(mr0, vi);
      tmpv3 = _mm_mul_pd(mi0, vr);
      _mm_store_pd(tmp, _mm_sub_pd(tmpv0, tmpv1));
      _mm_store_pd(tmp + 2, _mm_add_pd(tmpv2, tmpv3));
      data[inds[0]].real(tmp[0] + tmp[1]);
      data[inds[0]].imag(tmp[2] + tmp[3]);

      //
      // second row
      //
      // real part
      tmpv0 = _mm_mul_pd(mr1, vr);
      tmpv1 = _mm_mul_pd(mi1, vi);
      // imag part
      tmpv2 = _mm_mul_pd(mr1, vi);
      tmpv3 = _mm_mul_pd(mi1, vr);
      _mm_store_pd(tmp, _mm_sub_pd(tmpv0, tmpv1));
      _mm_store_pd(tmp + 2, _mm_add_pd(tmpv2, tmpv3));
      data[inds[1]].real(tmp[0] + tmp[1]);
      data[inds[1]].imag(tmp[2] + tmp[3]);
    }
  }

  inline void B2F32(size_t q_num, DType *data,
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

    alignas(16) float mptr[32];
    assert((((size_t)mptr) & 0b1111) == 0);  // Check 16 bytes alignment;
    mptr[0] = gate[0].real();
    mptr[1] = gate[1].real();
    mptr[2] = gate[2].real();
    mptr[3] = gate[3].real();
    mptr[4] = gate[4].real();
    mptr[5] = gate[5].real();
    mptr[6] = gate[6].real();
    mptr[7] = gate[7].real();
    mptr[8] = gate[8].real();
    mptr[9] = gate[9].real();
    mptr[10] = gate[10].real();
    mptr[11] = gate[11].real();
    mptr[12] = gate[12].real();
    mptr[13] = gate[13].real();
    mptr[14] = gate[14].real();
    mptr[15] = gate[15].real();
    mptr[16] = gate[0].imag();
    mptr[17] = gate[1].imag();
    mptr[18] = gate[2].imag();
    mptr[19] = gate[3].imag();
    mptr[20] = gate[4].imag();
    mptr[21] = gate[5].imag();
    mptr[22] = gate[6].imag();
    mptr[23] = gate[7].imag();
    mptr[24] = gate[8].imag();
    mptr[25] = gate[9].imag();
    mptr[26] = gate[10].imag();
    mptr[27] = gate[11].imag();
    mptr[28] = gate[12].imag();
    mptr[29] = gate[13].imag();
    mptr[30] = gate[14].imag();
    mptr[31] = gate[15].imag();

#pragma omp parallel for nowait
    for (size_t iter = 0; iter < iter_cnt; ++iter) {
      size_t base_ind =
          ((iter & mask2) << 2) | ((iter & mask1) << 1) | (iter & mask0);
      size_t inds[4];
      inds[0] = base_ind;
      inds[1] = inds[0] | (1LL << pos1);
      inds[2] = inds[0] | (1LL << pos0);
      inds[3] = inds[1] | (1LL << pos0);

      alignas(16) float tmp[8];
      assert((((size_t)tmp) & 0b1111) == 0);  // Check 16 bytes alignment;
      alignas(16) float vptr[8];
      assert((((size_t)vptr) & 0b1111) == 0);  // Check 16 bytes alignment;
      vptr[0] = data[inds[0]].real();
      vptr[1] = data[inds[1]].real();
      vptr[2] = data[inds[2]].real();
      vptr[3] = data[inds[3]].real();
      vptr[4] = data[inds[0]].imag();
      vptr[5] = data[inds[1]].imag();
      vptr[6] = data[inds[2]].imag();
      vptr[7] = data[inds[3]].imag();

      // [vr0, vr1, vr2, vr3]
      __m128 vr = _mm_load_ps(vptr);
      // [vi0, vi1, vi2, vi3]
      __m128 vi = _mm_load_ps(vptr + 4);
      __m128 mr, mi;
      __m128 tmpv0, tmpv1, tmpv2, tmpv3;

#define ONE_ROW_OP(offset, data_pos)                        \
  /* [mr?0, mr?1, mr?2, mr?3] */                            \
  mr = _mm_load_ps(mptr + (offset));                        \
  /* [mi?0, mi?1, mi?2, mi?3] */                            \
  mi = _mm_load_ps(mptr + (offset) + 16);                   \
                                                            \
  tmpv0 = _mm_mul_ps(mr, vr);                               \
  tmpv1 = _mm_mul_ps(mi, vi);                               \
  tmpv2 = _mm_mul_ps(mr, vi);                               \
  tmpv3 = _mm_mul_ps(mi, vr);                               \
  /* real part */                                           \
  _mm_store_ps(tmp, _mm_sub_ps(tmpv0, tmpv1));              \
  /* imag part */                                           \
  _mm_store_ps(tmp + 4, _mm_add_ps(tmpv2, tmpv3));          \
  data[(data_pos)].real(tmp[0] + tmp[1] + tmp[2] + tmp[3]); \
  data[(data_pos)].imag(tmp[4] + tmp[5] + tmp[6] + tmp[7]);

      // 1st row
      ONE_ROW_OP(0, inds[0]);
      // 2nd row
      ONE_ROW_OP(4, inds[1]);
      // 3rd row
      ONE_ROW_OP(8, inds[2]);
      // 4th row
      ONE_ROW_OP(12, inds[3]);

#undef ONE_ROW_OP
    }
  }

  inline void B2F64(size_t q_num, DType *data,
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

    alignas(16) double mptr[32];
    assert((((size_t)mptr) & 0b1111) == 0);  // Check 16 bytes alignment;
    mptr[0] = gate[0].real();
    mptr[1] = gate[1].real();
    mptr[2] = gate[2].real();
    mptr[3] = gate[3].real();
    mptr[4] = gate[4].real();
    mptr[5] = gate[5].real();
    mptr[6] = gate[6].real();
    mptr[7] = gate[7].real();
    mptr[8] = gate[8].real();
    mptr[9] = gate[9].real();
    mptr[10] = gate[10].real();
    mptr[11] = gate[11].real();
    mptr[12] = gate[12].real();
    mptr[13] = gate[13].real();
    mptr[14] = gate[14].real();
    mptr[15] = gate[15].real();
    mptr[16] = gate[0].imag();
    mptr[17] = gate[1].imag();
    mptr[18] = gate[2].imag();
    mptr[19] = gate[3].imag();
    mptr[20] = gate[4].imag();
    mptr[21] = gate[5].imag();
    mptr[22] = gate[6].imag();
    mptr[23] = gate[7].imag();
    mptr[24] = gate[8].imag();
    mptr[25] = gate[9].imag();
    mptr[26] = gate[10].imag();
    mptr[27] = gate[11].imag();
    mptr[28] = gate[12].imag();
    mptr[29] = gate[13].imag();
    mptr[30] = gate[14].imag();
    mptr[31] = gate[15].imag();

#pragma omp parallel for nowait
    for (size_t iter = 0; iter < iter_cnt; ++iter) {
      size_t base_ind =
          ((iter & mask2) << 2) | ((iter & mask1) << 1) | (iter & mask0);
      size_t inds[4];
      inds[0] = base_ind;
      inds[1] = inds[0] | (1LL << pos1);
      inds[2] = inds[0] | (1LL << pos0);
      inds[3] = inds[1] | (1LL << pos0);

      alignas(16) double tmp[4];
      assert((((size_t)tmp) & 0b1111) == 0);  // Check 16 bytes alignment;
      alignas(16) double vptr[8];
      assert((((size_t)vptr) & 0b1111) == 0);  // Check 16 bytes alignment;
      vptr[0] = data[inds[0]].real();
      vptr[1] = data[inds[1]].real();
      vptr[2] = data[inds[2]].real();
      vptr[3] = data[inds[3]].real();
      vptr[4] = data[inds[0]].imag();
      vptr[5] = data[inds[1]].imag();
      vptr[6] = data[inds[2]].imag();
      vptr[7] = data[inds[3]].imag();

      // [vr0, vr1]
      __m128d vra = _mm_load_pd(vptr);
      // [vr2, vr3]
      __m128d vrb = _mm_load_pd(vptr + 2);
      // [vi0, vi1]
      __m128d via = _mm_load_pd(vptr + 4);
      // [vi2, vi3]
      __m128d vib = _mm_load_pd(vptr + 6);

      __m128d tmpv0, tmpv1, tmpv2, tmpv3;
      __m128d mra, mrb, mia, mib;

#define ONE_ROW_OP(offset, data_pos)                              \
  /* [mr?0, mr?1] */                                              \
  mra = _mm_load_pd(mptr + (offset));                             \
  /* [mr?2, mr?3] */                                              \
  mrb = _mm_load_pd(mptr + (offset) + 2);                         \
  /* [mi?0, mi?1] */                                              \
  mia = _mm_load_pd(mptr + (offset) + 16);                        \
  /* [mi?2, mi?3] */                                              \
  mib = _mm_load_pd(mptr + (offset) + 18);                        \
                                                                  \
  /* real part */                                                 \
  tmpv0 = _mm_add_pd(_mm_mul_pd(mra, vra), _mm_mul_pd(mrb, vrb)); \
  tmpv1 = _mm_add_pd(_mm_mul_pd(mia, via), _mm_mul_pd(mib, vib)); \
  /* imag part */                                                 \
  tmpv2 = _mm_add_pd(_mm_mul_pd(mra, via), _mm_mul_pd(mrb, vib)); \
  tmpv3 = _mm_add_pd(_mm_mul_pd(mia, vra), _mm_mul_pd(mib, vrb)); \
  _mm_store_pd(tmp, _mm_sub_pd(tmpv0, tmpv1));                    \
  _mm_store_pd(tmp + 2, _mm_add_pd(tmpv2, tmpv3));                \
  data[(data_pos)].real(tmp[0] + tmp[1]);                         \
  data[(data_pos)].imag(tmp[2] + tmp[3]);

      ONE_ROW_OP(0, inds[0]);
      ONE_ROW_OP(4, inds[1]);
      ONE_ROW_OP(8, inds[2]);
      ONE_ROW_OP(12, inds[3]);

#undef ONE_ROW_OP
    }
  }
};
}  // namespace sim

#endif

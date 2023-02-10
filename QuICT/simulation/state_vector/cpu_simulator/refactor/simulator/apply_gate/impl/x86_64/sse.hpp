#ifndef QUICT_SIM_BACKEND_APPLY_GATE_IMPL_X86_64_SSE_H
#define QUICT_SIM_BACKEND_APPLY_GATE_IMPL_X86_64_SSE_H

#include <immintrin.h>

#include <complex>
#include <stdexcept>
#include <type_traits>

#include "../../delegate.hpp"

namespace sim {
template <class DType>
class SseApplyGateDelegate : public ApplyGateDelegate<DType> {
 public:
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

    // SSE for normal unitary
#pragma omp parallel for nowait
    for (size_t iter = 0; iter < iter_cnt; ++iter) {
      size_t base_ind = ((iter & mask1) << 1) | (iter & mask0);
      size_t inds[2] = {base_ind, base_ind | (1LL << pos)};
      // Mat-vec complex mul (2x2, 2)
      float tmp[4];
      float *v_raw = (float *)(data);
      float vptr[4];
      vptr[0] = v_raw[inds[0]];
      vptr[1] = v_raw[inds[0] + 1];
      vptr[2] = v_raw[inds[1]];
      vptr[3] = v_raw[inds[1] + 1];

      float *mptr = (float *)(&gate[0]);

      //
      // first row
      //
      // [mr00, mi00, mr01, mi01]
      __m128 m = _mm_load_ps(mptr);
      // [vr0, vi0, vr1, vi1]
      __m128 vp = _mm_load_ps(vptr);
      // [-vr0, -vi0, -vr1, -vi1]
      __m128 vn = _mm_sub_ps(_mm_setzero_ps(), vp);
      // [vr0, vi0, -vr0, -vi0]
      __m128 v0 = _mm_shuffle_ps(vp, vn, 0b01000100);
      // [vr1, vi1, -vr1, -vi1]
      __m128 v1 = _mm_shuffle_ps(vp, vn, 0b11101110);
      // real part
      // [vr0, -vi0, vr1, -vi1]
      __m128 v = _mm_shuffle_ps(v0, v1, 0b11001100);
      _mm_store_ps(tmp, _mm_mul_ps(m, v));
      v_raw[inds[0]] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
      // imag part
      // [vi0, vr0, vi1, vr1]
      v = _mm_shuffle_ps(v0, v1, 0b00010001);
      _mm_store_ps(tmp, _mm_mul_ps(m, v));
      v_raw[inds[0] + 1] = tmp[0] + tmp[1] + tmp[2] + tmp[3];

      //
      // second row
      //
      // [mr10, mi10, mr11, mi11]
      m = _mm_load_ps(mptr + 4);
      // [vr0, vi0, vr1, vi1]
      vp = _mm_load_ps(vptr);
      // [-vr0, -vi0, -vr1, -vi1]
      vn = _mm_sub_ps(_mm_setzero_ps(), vp);
      // [vr0, vi0, -vr0, -vi0]
      v0 = _mm_shuffle_ps(vp, vn, 0b01000100);
      // [vr1, vi1, -vr1, -vi1]
      v1 = _mm_shuffle_ps(vp, vn, 0b11101110);
      // real part
      // [vr0, -vi0, vr1, -vi1]
      v = _mm_shuffle_ps(v0, v1, 0b11001100);
      _mm_store_ps(tmp, _mm_mul_ps(m, v));
      v_raw[inds[0]] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
      // imag part
      // [vi0, vr0, vi1, vr1]
      v = _mm_shuffle_ps(v0, v1, 0b00010001);
      _mm_store_ps(tmp, _mm_mul_ps(m, v));
      v_raw[inds[0] + 1] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
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

    // SSE for normal unitary
#pragma omp parallel for nowait
    for (size_t iter = 0; iter < iter_cnt; ++iter) {
      size_t base_ind = ((iter & mask1) << 1) | (iter & mask0);
      size_t inds[2] = {base_ind, base_ind | (1LL << pos)};
      // Mat-vec complex mul (2x2, 2):
      double tmp[2];
      double *v_raw = (double *)(data);
      double vptr[4];
      vptr[0] = v_raw[inds[0]];
      vptr[1] = v_raw[inds[0] + 1];
      vptr[2] = v_raw[inds[1]];
      vptr[3] = v_raw[inds[1] + 1];
      double *mptr = (double *)(&gate[0]);
      // [vr0, vi0]
      __m128d v0 = _mm_load_pd(vptr);
      // [vr1, vi1]
      __m128d v1 = _mm_load_pd(vptr + 2);
      // [vr0, vr1]
      __m128d vr = _mm_shuffle_pd(v0, v1, 0b00);
      // [vi0, vi1]
      __m128d vi = _mm_shuffle_pd(v0, v1, 0b11);

      //
      // first row
      //
      // [mr00, mi00]
      v0 = _mm_load_pd(mptr);
      // [mr01, mi01]
      v1 = _mm_load_pd(mptr + 2);
      // [mr00, mr01]
      __m128d mr = _mm_shuffle_pd(v0, v1, 0b00);
      // [mi00, mi01]
      __m128d mi = _mm_shuffle_pd(v0, v1, 0b11);
      // real part
      v0 = _mm_mul_pd(mr, vr);
      v1 = _mm_mul_pd(mi, vi);
      _mm_store_pd(tmp, _mm_sub_pd(v0, v1));
      v_raw[inds[0]] = tmp[0] + tmp[1];
      // imag part
      v0 = _mm_mul_pd(mr, vi);
      v1 = _mm_mul_pd(mi, vr);
      _mm_store_pd(tmp, _mm_add_pd(v0, v1));
      v_raw[inds[0] + 1] = tmp[0] + tmp[1];

      //
      // second row
      //
      // [mr10, mi10]
      v0 = _mm_load_pd(mptr + 4);
      // [mr11, mi11]
      v1 = _mm_load_pd(mptr + 6);
      // [mr10, mr11]
      mr = _mm_shuffle_pd(v0, v1, 0b00);
      // [mi10, mi11]
      mi = _mm_shuffle_pd(v0, v1, 0b11);
      // real part
      v0 = _mm_mul_pd(mr, vr);
      v1 = _mm_mul_pd(mi, vi);
      _mm_store_pd(tmp, _mm_sub_pd(v0, v1));
      v_raw[inds[1]] = tmp[0] + tmp[1];
      // imag part
      v0 = _mm_mul_pd(mr, vi);
      v1 = _mm_mul_pd(mi, vr);
      _mm_store_pd(tmp, _mm_add_pd(v0, v1));
      v_raw[inds[1] + 1] = tmp[0] + tmp[1];
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

#pragma omp parallel for nowait
    for (size_t iter = 0; iter < iter_cnt; ++iter) {
      size_t base_ind =
          ((iter & mask2) << 2) | ((iter & mask1) << 1) | (iter & mask0);
      size_t inds[4];
      inds[0] = base_ind;
      inds[1] = inds[0] | (1LL << pos1);
      inds[2] = inds[0] | (1LL << pos0);
      inds[3] = inds[1] | (1LL << pos0);

      float tmp[4];
      float *mptr = (float *)(&gate[0]);
      float *v_raw = (float *)(data);
      float vptr[8];
      vptr[0] = v_raw[inds[0]];
      vptr[1] = v_raw[inds[0] + 1];
      vptr[2] = v_raw[inds[1]];
      vptr[3] = v_raw[inds[1] + 1];
      vptr[4] = v_raw[inds[2]];
      vptr[5] = v_raw[inds[2] + 1];
      vptr[6] = v_raw[inds[3]];
      vptr[7] = v_raw[inds[3] + 1];

      // [vr0, vi0, vr1, vi1]
      __m128d tmpv0 = _mm_load_ps(vptr);
      // [vr2, vi2, vr3, vi3]
      __m128d tmpv1 = _mm_load_ps(vptr + 4);
      // [vr0, vr1, vr2, vr3]
      __m128d vr = _mm_shuffle_ps(tmpv0, tmpv1, 0b10001000);
      // [vi0, vi1, vi2, vi3]
      __m128d vi = _mm_shuffle_ps(tmpv0, tmpv1, 0b11101110);
      __m128d mr, mi;

#define ONE_ROW_OP(offset, data_pos)                     \
  /* [mr?0, mi?0, mr?1, mi?1] */                         \
  tmpv0 = _mm_load_ps(mptr + (offset));                  \
  /* [mr?2, mi?2, mr?3, mi?3] */                         \
  tmpv1 = _mm_load_ps(mptr + (offset) + 4);              \
  /* [mr?0, mr?1, mr?2, mr?3] */                         \
  mr = _mm_shuffle_ps(tmpv0, tmpv1, 0b10001000);         \
  /* [mi?0, mi?1, mi?2, mi?3] */                         \
  mi = _mm_shuffle_ps(tmpv0, tmpv1, 0b11101110);         \
                                                         \
  /* real part */                                        \
  tmpv0 = _mm_mul_ps(mr, vr);                            \
  tmpv1 = _mm_mul_ps(mi, vi);                            \
  _mm_store_ps(tmp, _mm_sub_ps(tmpv0, tmpv1));           \
  v_raw[(data_pos)] = tmp[0] + tmp[1] + tmp[2] + tmp[3]; \
                                                         \
  /* imag part */                                        \
  tmpv0 = _mm_mul_ps(mr, vi);                            \
  tmpv1 = _mm_mul_ps(mi, vr);                            \
  _mm_store_ps(tmp, _mm_add_ps(tmpv0, tmpv1));           \
  v_raw[(data_pos) + 1] = tmp[0] + tmp[1] + tmp[2] + tmp[3];

      // 1st row
      ONE_ROW_OP(0, inds[0]);
      // 2nd row
      ONE_ROW_OP(8, inds[1]);
      // 3rd row
      ONE_ROW_OP(16, inds[2]);
      // 4th row
      ONE_ROW_OP(24, inds[3]);

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

    // SSE for unitary
#pragma omp parallel for nowait
    for (size_t iter = 0; iter < iter_cnt; ++iter) {
      size_t base_ind =
          ((iter & mask2) << 2) | ((iter & mask1) << 1) | (iter & mask0);
      size_t inds[4];
      inds[0] = base_ind;
      inds[1] = inds[0] | (1LL << pos1);
      inds[2] = inds[0] | (1LL << pos0);
      inds[3] = inds[1] | (1LL << pos0);

      double tmp[4];
      double *mptr = (double *)(&gate[0]);
      double *v_raw = (double *)(data);
      double vptr[8];
      vptr[0] = v_raw[inds[0]];
      vptr[1] = v_raw[inds[0] + 1];
      vptr[2] = v_raw[inds[1]];
      vptr[3] = v_raw[inds[1] + 1];
      vptr[4] = v_raw[inds[2]];
      vptr[5] = v_raw[inds[2] + 1];
      vptr[6] = v_raw[inds[3]];
      vptr[7] = v_raw[inds[3] + 1];

      // On x86_64 architecture, there are 16 XMM registers available.
      // So don't worry about register usage. :)
      // https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions

      // [vr0, vi0]
      __m128d tmpv0 = _mm_load_pd(vptr);
      // [vr1, vi1]
      __m128d tmpv1 = _mm_load_pd(vptr + 2);
      // [vr0, vr1]
      __m128d vra = _mm_shuffle_pd(tmpv0, tmpv1, 0b00);
      // [vi0, vi1]
      __m128d via = _mm_shuffle_pd(tmpv0, tmpv1, 0b11);
      // [vr2, vi2]
      tmpv0 = _mm_load_pd(vptr + 4);
      // [vr3, vi3]
      tmpv1 = _mm_load_pd(vptr + 6);
      // [vr2, vr3]
      __m128d vrb = _mm_shuffle_pd(tmpv0, tmpv1, 0b00);
      // [vi2, vi3]
      __m128d vib = _mm_shuffle_pd(tmpv0, tmpv1, 0b11);
      __m128d mra, mrb, mia, mib;
#define ONE_ROW_OP(offset, data_pos)                              \
  /* [mr?0, mi?0] */                                              \
  tmpv0 = _mm_load_pd(mptr + (offset));                           \
  /* [mr?1, mi?1] */                                              \
  tmpv1 = _mm_load_pd(mptr + (offset) + 2);                       \
  /* [mr?0, mr?1] */                                              \
  mra = _mm_shuffle_pd(tmpv0, tmpv1, 0b00);                       \
  mia = _mm_shuffle_pd(tmpv0, tmpv1, 0b11);                       \
  tmpv0 = _mm_load_pd(mptr + (offset) + 4);                       \
  tmpv1 = _mm_load_pd(mptr + (offset) + 6);                       \
  /* [mr?2, mr?3] */                                              \
  mrb = _mm_shuffle_pd(tmpv0, tmpv1, 0b00);                       \
  /* [mi?2, mi?3] */                                              \
  mib = _mm_shuffle_pd(tmpv0, tmpv1, 0b11);                       \
                                                                  \
  /* real part */                                                 \
  tmpv0 = _mm_add_pd(_mm_mul_pd(mra, vra), _mm_mul_pd(mrb, vrb)); \
  tmpv1 = _mm_add_pd(_mm_mul_pd(mia, via), _mm_mul_pd(mib, vib)); \
  _mm_store_pd(tmp, _mm_sub_pd(tmpv0, tmpv1));                    \
  v_raw[(data_pos)] = tmp[0] + tmp[1];                            \
                                                                  \
  /* imag part */                                                 \
  tmpv0 = _mm_add_pd(_mm_mul_pd(mra, via), _mm_mul_pd(mrb, vib)); \
  tmpv1 = _mm_add_pd(_mm_mul_pd(mia, vra), _mm_mul_pd(mib, vrb)); \
  _mm_store_pd(tmp, _mm_add_pd(tmpv0, tmpv1));                    \
  v_raw[(data_pos) + 1] = tmp[0] + tmp[1];

      ONE_ROW_OP(0, inds[0]);
      ONE_ROW_OP(8, inds[1]);
      ONE_ROW_OP(16, inds[2]);
      ONE_ROW_OP(24, inds[3]);

#undef ONE_ROW_OP
    }
  }
};
}  // namespace sim

#endif

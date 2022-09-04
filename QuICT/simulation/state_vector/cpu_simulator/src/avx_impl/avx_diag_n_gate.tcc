//
// Created by Ci Lei on 2021-11-05.
//

#ifndef SIM_BACK_AVX_DIAG_N_GATE_TCC
#define SIM_BACK_AVX_DIAG_N_GATE_TCC

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "../gate.h"
#include "../matricks_simulator.h"
#include "../utility.h"

namespace QuICT {
template <typename Precision>
template <uint64_t N, template <uint64_t, typename> class Gate>
void MaTricksSimulator<Precision>::apply_diag_n_gate(
    uint64_t q_state_bit_num, const Gate<N, Precision> &gate, Precision *real,
    Precision *imag, uint32_t omp_thread_num) {
  if constexpr (std::is_same_v<Precision, float>) {
    throw std::runtime_error(std::string(__FILE__) + ":" +
                             std::to_string(__LINE__) + ": " +
                             "Not Implemented " + __func__);
  } else if constexpr (std::is_same_v<Precision, double>) {
    if constexpr (N == 2) {
      uint64_t task_num = 1ULL << (q_state_bit_num - 2);

      uarray_t<2> qubits = {gate.affect_args_[0], gate.affect_args_[1]};
      uarray_t<2> qubits_sorted;
      if (gate.affect_args_[0] < gate.affect_args_[1]) {
        qubits_sorted[0] = gate.affect_args_[0];
        qubits_sorted[1] = gate.affect_args_[1];
      } else {
        qubits_sorted[1] = gate.affect_args_[0];
        qubits_sorted[0] = gate.affect_args_[1];
      }

      if (qubits_sorted[1] == q_state_bit_num - 1) {            // ...q
        if (qubits_sorted[0] == q_state_bit_num - 2) {          // ...qq
          __m256d ymm0 = _mm256_loadu_pd(gate.diagonal_real_);  // d_r
          __m256d ymm1 = _mm256_loadu_pd(gate.diagonal_imag_);  // d_i

          if (qubits_sorted[0] == qubits[0]) {  // ...01
            // v00 v01 v02 v03 v10 v11 v12 v13

          } else {  // ...10
            // v00 v02 v01 v03 v10 v12 v11 v13
            ymm0 = _mm256_permute4x64_pd(ymm0, 0b1101'1000);
            ymm1 = _mm256_permute4x64_pd(ymm1, 0b1101'1000);
          }

          constexpr uint64_t batch_size = 2;
#pragma omp parallel for num_threads(omp_thread_num) \
    schedule(dynamic, omp_chunk_size(q_state_bit_num))
          for (uint64_t task_id = 0; task_id < task_num;
               task_id += batch_size) {
            auto ind0 = index0(task_id, q_state_bit_num, qubits, qubits_sorted);
            __m256d ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9;
            ymm2 = _mm256_loadu_pd(&real[ind0]);
            ymm3 = _mm256_loadu_pd(&imag[ind0]);
            ymm4 = _mm256_loadu_pd(&real[ind0 + 4]);
            ymm5 = _mm256_loadu_pd(&imag[ind0 + 4]);

            COMPLEX_YMM_MUL(ymm0, ymm1, ymm2, ymm3, ymm6, ymm7);
            COMPLEX_YMM_MUL(ymm0, ymm1, ymm4, ymm5, ymm8, ymm9);

            _mm256_storeu_pd(&real[ind0], ymm6);
            _mm256_storeu_pd(&imag[ind0], ymm7);
            _mm256_storeu_pd(&real[ind0 + 4], ymm8);
            _mm256_storeu_pd(&imag[ind0 + 4], ymm9);
          }
        } else {  // ...q.q
          __m256d ymm0, ymm1, ymm2, ymm3;
          __m256d ymm12 = _mm256_loadu_pd(gate.diagonal_real_);
          __m256d ymm13 = _mm256_loadu_pd(gate.diagonal_imag_);
          if (qubits_sorted[0] == qubits[0]) {  // ...0.1
            // v00 v01 v10 v11 . v02 v03 v12 v13
          } else {  // ...1.0
            // v00 v02 v10 v12 . v01 v03 v11 v13
            ymm12 = _mm256_permute4x64_pd(ymm12, 0b1101'1000);
            ymm13 = _mm256_permute4x64_pd(ymm13, 0b1101'1000);
          }
          ymm0 = _mm256_permute2f128_pd(ymm12, ymm12, 0b0000'0000);
          ymm1 = _mm256_permute2f128_pd(ymm13, ymm13, 0b0000'0000);
          ymm2 = _mm256_permute2f128_pd(ymm12, ymm12, 0b0001'0001);
          ymm3 = _mm256_permute2f128_pd(ymm13, ymm13, 0b0001'0001);
          constexpr uint64_t batch_size = 2;
#pragma omp parallel for num_threads(omp_thread_num) \
    schedule(dynamic, omp_chunk_size(q_state_bit_num))
          for (uint64_t task_id = 0; task_id < task_num;
               task_id += batch_size) {
            auto inds = index(task_id, q_state_bit_num, qubits, qubits_sorted);
            __m256d ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11;
            if (qubits_sorted[0] == qubits[0]) {  // ...0.1
              // v00 v01 v10 v11 . v02 v03 v12 v13
              ymm4 = _mm256_loadu_pd(&real[inds[0]]);
              ymm5 = _mm256_loadu_pd(&imag[inds[0]]);
              ymm6 = _mm256_loadu_pd(&real[inds[2]]);
              ymm7 = _mm256_loadu_pd(&imag[inds[2]]);
            } else {  // ...1.0
              // v00 v02 v10 v12 . v01 v03 v11 v13
              ymm4 = _mm256_loadu_pd(&real[inds[0]]);
              ymm5 = _mm256_loadu_pd(&imag[inds[0]]);
              ymm6 = _mm256_loadu_pd(&real[inds[1]]);
              ymm7 = _mm256_loadu_pd(&imag[inds[1]]);
            }
            COMPLEX_YMM_MUL(ymm0, ymm1, ymm4, ymm5, ymm8, ymm9);
            COMPLEX_YMM_MUL(ymm2, ymm3, ymm6, ymm7, ymm10, ymm11);

            if (qubits_sorted[0] == qubits[0]) {  // ...0.1
              // v00 v01 v10 v11 . v02 v03 v12 v13
              _mm256_storeu_pd(&real[inds[0]], ymm8);
              _mm256_storeu_pd(&imag[inds[0]], ymm9);
              _mm256_storeu_pd(&real[inds[2]], ymm10);
              _mm256_storeu_pd(&imag[inds[2]], ymm11);
            } else {  // ...1.0
              // v00 v02 v10 v12 . v01 v03 v11 v13
              _mm256_storeu_pd(&real[inds[0]], ymm8);
              _mm256_storeu_pd(&imag[inds[0]], ymm9);
              _mm256_storeu_pd(&real[inds[1]], ymm10);
              _mm256_storeu_pd(&imag[inds[1]], ymm11);
            }
          }
        }
      } else if (qubits_sorted[1] == q_state_bit_num - 2) {  // ...q.
        __m256d ymm0, ymm1, ymm2, ymm3;
        if (qubits_sorted[0] == qubits[0]) {  // ...0.
          // v00 v10 v01 v11 ... v02 v12 v03 v13
          ymm0 = _mm256_setr_pd(gate.diagonal_real_[0], gate.diagonal_real_[0],
                                gate.diagonal_real_[1], gate.diagonal_real_[1]);
          ymm1 = _mm256_setr_pd(gate.diagonal_imag_[0], gate.diagonal_imag_[0],
                                gate.diagonal_imag_[1], gate.diagonal_imag_[1]);
          ymm2 = _mm256_setr_pd(gate.diagonal_real_[2], gate.diagonal_real_[2],
                                gate.diagonal_real_[3], gate.diagonal_real_[3]);
          ymm3 = _mm256_setr_pd(gate.diagonal_imag_[2], gate.diagonal_imag_[2],
                                gate.diagonal_imag_[3], gate.diagonal_imag_[3]);
        } else {  // ...1.
          // v00 v10 v02 v12 ... v01 v11 v03 v13
          ymm0 = _mm256_setr_pd(gate.diagonal_real_[0], gate.diagonal_real_[0],
                                gate.diagonal_real_[2], gate.diagonal_real_[2]);
          ymm1 = _mm256_setr_pd(gate.diagonal_imag_[0], gate.diagonal_imag_[0],
                                gate.diagonal_imag_[2], gate.diagonal_imag_[2]);
          ymm2 = _mm256_setr_pd(gate.diagonal_real_[1], gate.diagonal_real_[1],
                                gate.diagonal_real_[3], gate.diagonal_real_[3]);
          ymm3 = _mm256_setr_pd(gate.diagonal_imag_[1], gate.diagonal_imag_[1],
                                gate.diagonal_imag_[3], gate.diagonal_imag_[3]);
        }

        constexpr uint64_t batch_size = 2;
#pragma omp parallel for num_threads(omp_thread_num) \
    schedule(dynamic, omp_chunk_size(q_state_bit_num))
        for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
          auto inds = index(task_id, q_state_bit_num, qubits, qubits_sorted);
          __m256d ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11;
          if (qubits_sorted[0] == qubits[0]) {  // ...0.1
            // v00 v10 v01 v11 ... v02 v12 v03 v13
            ymm4 = _mm256_loadu_pd(&real[inds[0]]);
            ymm5 = _mm256_loadu_pd(&imag[inds[0]]);
            ymm6 = _mm256_loadu_pd(&real[inds[2]]);
            ymm7 = _mm256_loadu_pd(&imag[inds[2]]);
          } else {  // ...1.0
            // v00 v10 v02 v12 ... v01 v11 v03 v13
            ymm4 = _mm256_loadu_pd(&real[inds[0]]);
            ymm5 = _mm256_loadu_pd(&imag[inds[0]]);
            ymm6 = _mm256_loadu_pd(&real[inds[1]]);
            ymm7 = _mm256_loadu_pd(&imag[inds[1]]);
          }
          COMPLEX_YMM_MUL(ymm0, ymm1, ymm4, ymm5, ymm8, ymm9);
          COMPLEX_YMM_MUL(ymm2, ymm3, ymm6, ymm7, ymm10, ymm11);

          if (qubits_sorted[0] == qubits[0]) {  // ...0.1
            // v00 v10 v01 v11 ... v02 v12 v03 v13
            _mm256_storeu_pd(&real[inds[0]], ymm8);
            _mm256_storeu_pd(&imag[inds[0]], ymm9);
            _mm256_storeu_pd(&real[inds[2]], ymm10);
            _mm256_storeu_pd(&imag[inds[2]], ymm11);
          } else {  // ...1.0
            // v00 v10 v02 v12 ... v01 v11 v03 v13
            _mm256_storeu_pd(&real[inds[0]], ymm8);
            _mm256_storeu_pd(&imag[inds[0]], ymm9);
            _mm256_storeu_pd(&real[inds[1]], ymm10);
            _mm256_storeu_pd(&imag[inds[1]], ymm11);
          }
        }
      } else {  // xxx..
        // There are only 16 ymm registers.
        constexpr uint64_t batch_size = 4;
#pragma omp parallel for num_threads(omp_thread_num) \
    schedule(dynamic, omp_chunk_size(q_state_bit_num))
        for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
          auto inds = index(task_id, q_state_bit_num, qubits, qubits_sorted);
          for (int i = 0; i < 4; ++i) {
            __m256d ymm0 = _mm256_broadcast_sd(&gate.diagonal_real_[i]);
            __m256d ymm1 = _mm256_broadcast_sd(&gate.diagonal_imag_[i]);
            __m256d ymm2 = _mm256_loadu_pd(&real[inds[i]]);
            __m256d ymm3 = _mm256_loadu_pd(&imag[inds[i]]);
            __m256d ymm4, ymm5;
            COMPLEX_YMM_MUL(ymm0, ymm1, ymm2, ymm3, ymm4, ymm5);
            _mm256_storeu_pd(&real[inds[i]], ymm4);
            _mm256_storeu_pd(&imag[inds[i]], ymm5);
          }
        }
      }
    } else {  // N == 1
      uint64_t task_num = 1ULL << (q_state_bit_num - 1);
      if (gate.targ_ == q_state_bit_num - 1) {
        __m256d ymm0 = _mm256_loadu2_m128d(gate.diagonal_real_,
                                           gate.diagonal_real_);  // d_r
        __m256d ymm1 = _mm256_loadu2_m128d(gate.diagonal_imag_,
                                           gate.diagonal_imag_);  // d_i
        constexpr uint64_t batch_size = 2;
#pragma omp parallel for num_threads(omp_thread_num) \
    schedule(dynamic, omp_chunk_size(q_state_bit_num))
        for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
          auto ind0 = index0(task_id, q_state_bit_num, gate.targ_);
          __m256d ymm2 = _mm256_loadu_pd(&real[ind0]);  // v_r
          __m256d ymm3 = _mm256_loadu_pd(&imag[ind0]);  // v_i
          __m256d ymm4, ymm5;
          COMPLEX_YMM_MUL(ymm0, ymm1, ymm2, ymm3, ymm4, ymm5);
          _mm256_storeu_pd(&real[ind0], ymm4);
          _mm256_storeu_pd(&imag[ind0], ymm5);
        }
      } else if (gate.targ_ == q_state_bit_num - 2) {
        __m256d ymm0 =
            _mm256_loadu2_m128d(gate.diagonal_real_, gate.diagonal_real_);
        __m256d ymm1 =
            _mm256_loadu2_m128d(gate.diagonal_imag_, gate.diagonal_imag_);
        ymm0 = _mm256_permute4x64_pd(ymm0, 0b1101'1000);  // d_r
        ymm1 = _mm256_permute4x64_pd(ymm1, 0b1101'1000);  // d_i
        constexpr uint64_t batch_size = 2;
#pragma omp parallel for num_threads(omp_thread_num) \
    schedule(dynamic, omp_chunk_size(q_state_bit_num))
        for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
          auto ind0 = index0(task_id, q_state_bit_num, gate.targ_);
          __m256d ymm2 = _mm256_loadu_pd(&real[ind0]);  // v_r
          __m256d ymm3 = _mm256_loadu_pd(&imag[ind0]);  // v_i
          __m256d ymm4, ymm5;
          COMPLEX_YMM_MUL(ymm0, ymm1, ymm2, ymm3, ymm4, ymm5);
          _mm256_storeu_pd(&real[ind0], ymm4);
          _mm256_storeu_pd(&imag[ind0], ymm5);
        }
      } else {  // gate.targ_ < q_state_bit_num - 2
        __m256d ymm0 = _mm256_broadcast_sd(&gate.diagonal_real_[0]);
        __m256d ymm1 = _mm256_broadcast_sd(&gate.diagonal_imag_[0]);
        __m256d ymm2 = _mm256_broadcast_sd(&gate.diagonal_real_[1]);
        __m256d ymm3 = _mm256_broadcast_sd(&gate.diagonal_imag_[1]);
        constexpr uint64_t batch_size = 4;
#pragma omp parallel for num_threads(omp_thread_num) \
    schedule(dynamic, omp_chunk_size(q_state_bit_num))
        for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
          auto inds = index(task_id, q_state_bit_num, gate.targ_);
          __m256d ymm4 =
              _mm256_loadu_pd(&real[inds[0]]);  // v00 v10 v20 v30, real
          __m256d ymm5 =
              _mm256_loadu_pd(&imag[inds[0]]);  // v00 v10 v20 v30, imag
          __m256d ymm6 =
              _mm256_loadu_pd(&real[inds[1]]);  // v01 v11 v21 v31, real
          __m256d ymm7 =
              _mm256_loadu_pd(&imag[inds[1]]);  // v01 v11 v21 v31, imag
          __m256d ymm8, ymm9, ymm10, ymm11;
          COMPLEX_YMM_MUL(ymm0, ymm1, ymm4, ymm5, ymm8, ymm9);
          COMPLEX_YMM_MUL(ymm2, ymm3, ymm6, ymm7, ymm10, ymm11);
          _mm256_storeu_pd(&real[inds[0]], ymm8);
          _mm256_storeu_pd(&imag[inds[0]], ymm9);
          _mm256_storeu_pd(&real[inds[1]], ymm10);
          _mm256_storeu_pd(&imag[inds[1]], ymm11);
        }
      }
    }
  }
}
}  // namespace QuICT

#endif  // SIM_BACK_AVX_DIAG_N_GATE_TCC

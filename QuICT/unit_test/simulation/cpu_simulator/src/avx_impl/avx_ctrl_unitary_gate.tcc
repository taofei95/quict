//
// Created by Ci Lei on 2021-11-05.
//

#ifndef SIM_BACK_AVX_CTRL_UNITARY_GATE_TCC
#define SIM_BACK_AVX_CTRL_UNITARY_GATE_TCC

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
template <template <typename> class Gate>
void MaTricksSimulator<Precision>::apply_ctrl_unitary_gate(
    uint64_t q_state_bit_num, const Gate<Precision> &gate, Precision *real,
    Precision *imag, uint32_t omp_thread_num) {
  if constexpr (std::is_same_v<Precision, float>) {
    throw std::runtime_error(std::string(__FILE__) + ":" +
                             std::to_string(__LINE__) + ": " +
                             "Not Implemented " + __func__);
  } else if constexpr (std::is_same_v<Precision, double>) {
    uarray_t<2> qubits = {gate.carg_, gate.targ_};
    uarray_t<2> qubits_sorted = {gate.carg_, gate.targ_};
    if (gate.carg_ > gate.targ_) {
      qubits_sorted[0] = gate.targ_;
      qubits_sorted[1] = gate.carg_;
    }

    /*
     * Multiplication Relation:
     * vi2 <-> m0/m2
     * vi3 <-> m1/m3
     * */

    uint64_t task_num = 1ULL << (q_state_bit_num - 2);
    if (qubits_sorted[1] == q_state_bit_num - 1) {
      if (qubits_sorted[0] == q_state_bit_num - 2) {
        __m256d ymm0, ymm1, ymm2, ymm3;
        ymm0 = _mm256_setr_pd(gate.mat_real_[0], gate.mat_real_[1],
                              gate.mat_real_[0],
                              gate.mat_real_[1]);  // m0 m1 m0 m1, real
        ymm1 = _mm256_setr_pd(gate.mat_imag_[0], gate.mat_imag_[1],
                              gate.mat_imag_[0],
                              gate.mat_imag_[1]);  // m0 m1 m0 m1, imag
        ymm2 = _mm256_setr_pd(gate.mat_real_[2], gate.mat_real_[3],
                              gate.mat_real_[2],
                              gate.mat_real_[3]);  // m2 m3 m2 m3, real
        ymm3 = _mm256_setr_pd(gate.mat_imag_[2], gate.mat_imag_[3],
                              gate.mat_imag_[2],
                              gate.mat_imag_[3]);  // m2 m3 m2 m3, imag

        constexpr uint64_t batch_size = 2;

#pragma omp parallel for num_threads(omp_thread_num) \
    schedule(dynamic, omp_chunk_size(q_state_bit_num))
        for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
          __m256d ymm4, ymm5, ymm6, ymm7, ymm8, ymm9;
          auto ind0 = index0(task_id, q_state_bit_num, qubits, qubits_sorted);
          if (qubits[0] == qubits_sorted[0]) {  // ...q0q1
            // v00 v01 v02 v03 v10 v11 v12 v13
            ymm4 = _mm256_loadu2_m128d(
                &real[ind0 + 6], &real[ind0 + 2]);  // v02 v03 v12 v13, real
            ymm5 = _mm256_loadu2_m128d(
                &imag[ind0 + 6], &imag[ind0 + 2]);  // v02 v03 v12 v13, imag
          } else {                                  // ...q1q0
            // v00 v02 v01 v03 v10 v12 v11 v13
            STRIDE_2_LOAD_ODD_PD(&real[ind0], ymm4, ymm6,
                                 ymm7);  // v02 v03 v12 v13, real
            STRIDE_2_LOAD_ODD_PD(&imag[ind0], ymm5, ymm6,
                                 ymm7);  // v02 v03 v12 v13, imag
          }
          // Now: ymm4 -> v_r, ymm5 -> v_i
          COMPLEX_YMM_MUL(ymm0, ymm1, ymm4, ymm5, ymm6, ymm7);
          COMPLEX_YMM_MUL(ymm2, ymm3, ymm4, ymm5, ymm8, ymm9);
          ymm4 = _mm256_hadd_pd(ymm6, ymm8);  // res_r
          ymm5 = _mm256_hadd_pd(ymm7, ymm9);  // res_i

          if (qubits[0] == qubits_sorted[0]) {  // ...q0q1
            // v00 v01 v02 v03 v10 v11 v12 v13
            _mm256_storeu2_m128d(&real[ind0 + 6], &real[ind0 + 2], ymm4);
            _mm256_storeu2_m128d(&imag[ind0 + 6], &imag[ind0 + 2], ymm5);
          } else {  // ...q1q0
            // v00 v02 v01 v03 v10 v12 v11 v13
            Precision tmp_r[4], tmp_i[4];
            STRIDE_2_STORE_ODD_PD(&real[ind0], ymm4, tmp_r);
            STRIDE_2_STORE_ODD_PD(&imag[ind0], ymm5, tmp_i);
          }
        }

      } else if (qubits_sorted[0] < q_state_bit_num - 2) {
        // Actually copied from above codes
        // Maybe we can eliminate duplications :(
        __m256d ymm0, ymm1, ymm2, ymm3;
        ymm0 = _mm256_setr_pd(gate.mat_real_[0], gate.mat_real_[1],
                              gate.mat_real_[0],
                              gate.mat_real_[1]);  // m0 m1 m0 m1, real
        ymm1 = _mm256_setr_pd(gate.mat_imag_[0], gate.mat_imag_[1],
                              gate.mat_imag_[0],
                              gate.mat_imag_[1]);  // m0 m1 m0 m1, imag
        ymm2 = _mm256_setr_pd(gate.mat_real_[2], gate.mat_real_[3],
                              gate.mat_real_[2],
                              gate.mat_real_[3]);  // m2 m3 m2 m3, real
        ymm3 = _mm256_setr_pd(gate.mat_imag_[2], gate.mat_imag_[3],
                              gate.mat_imag_[2],
                              gate.mat_imag_[3]);  // m2 m3 m2 m3, imag
        constexpr uint64_t batch_size = 2;

#pragma omp parallel for num_threads(omp_thread_num) \
    schedule(dynamic, omp_chunk_size(q_state_bit_num))
        for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
          __m256d ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11;
          auto inds = index(task_id, q_state_bit_num, qubits, qubits_sorted);
          if (qubits_sorted[0] == qubits[0]) {  // ...q0.q1
            // v00 v01 v10 v11 . v02 v03 v12 v13
            ymm4 = _mm256_loadu_pd(&real[inds[2]]);  // v_r
            ymm5 = _mm256_loadu_pd(&imag[inds[2]]);  // v_r
          } else {                                   // ...q1.q0
            // v00 v02 v10 v12 . v01 v03 v11 v13
            ymm6 = _mm256_loadu_pd(&real[inds[0]]);
            ymm7 = _mm256_loadu_pd(&real[inds[1]]);
            ymm8 = _mm256_loadu_pd(&imag[inds[0]]);
            ymm9 = _mm256_loadu_pd(&imag[inds[1]]);
            ymm4 =
                _mm256_shuffle_pd(ymm6, ymm7, 0b1111);  // v02 v03 v12 v13, real
            ymm5 =
                _mm256_shuffle_pd(ymm8, ymm9, 0b1111);  // v02 v03 v12 v13, imag
            ymm10 =
                _mm256_shuffle_pd(ymm6, ymm7, 0b0000);  // v00 v01 v10 v11, real
            ymm11 =
                _mm256_shuffle_pd(ymm8, ymm9, 0b0000);  // v00 v01 v10 v11, imag
          }
          // Now: ymm4 -> v_r, ymm5 -> v_i
          COMPLEX_YMM_MUL(ymm0, ymm1, ymm4, ymm5, ymm6, ymm7);
          COMPLEX_YMM_MUL(ymm2, ymm3, ymm4, ymm5, ymm8, ymm9);
          ymm4 = _mm256_hadd_pd(ymm6, ymm8);  // res_r
          ymm5 = _mm256_hadd_pd(ymm7, ymm9);  // res_i

          if (qubits_sorted[0] == qubits[0]) {  // ...q0.q1
            // v00 v01 v10 v11 . v02 v03 v12 v13
            _mm256_storeu_pd(&real[inds[2]], ymm4);
            _mm256_storeu_pd(&imag[inds[2]], ymm5);
          } else {  // ...q1.q0
            // v00 v02 v10 v12 . v01 v03 v11 v13
            ymm6 = _mm256_shuffle_pd(ymm10, ymm4,
                                     0b0000);  // v00 v02 v12 v12, real
            ymm7 = _mm256_shuffle_pd(ymm11, ymm5,
                                     0b0000);  // v00 v02 v12 v12, imag
            ymm8 = _mm256_shuffle_pd(ymm10, ymm4,
                                     0b1111);  // v01 v03 v11 v13, real
            ymm9 = _mm256_shuffle_pd(ymm11, ymm5,
                                     0b1111);  // v01 v03 v11 v13, imag
            _mm256_storeu_pd(&real[inds[0]], ymm6);
            _mm256_storeu_pd(&real[inds[1]], ymm8);
            _mm256_storeu_pd(&imag[inds[0]], ymm7);
            _mm256_storeu_pd(&imag[inds[1]], ymm9);
          }
        }
      }
    } else if (qubits_sorted[1] == q_state_bit_num - 2) {
      constexpr uint64_t batch_size = 2;
      __m256d ymm0 = _mm256_loadu2_m128d(
          &gate.mat_real_[0], &gate.mat_real_[0]);  // m0 m1 m0 m1, real
      __m256d ymm1 = _mm256_loadu2_m128d(
          &gate.mat_real_[2], &gate.mat_real_[2]);  // m2 m3 m2 m3, real
      __m256d ymm2 = _mm256_loadu2_m128d(
          &gate.mat_imag_[0], &gate.mat_imag_[0]);  // m0 m1 m0 m1, imag
      __m256d ymm3 = _mm256_loadu2_m128d(
          &gate.mat_imag_[2], &gate.mat_imag_[2]);  // m2 m3 m2 m3, imag

#pragma omp parallel for num_threads(omp_thread_num) \
    schedule(dynamic, omp_chunk_size(q_state_bit_num))
      for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
        __m256d ymm4, ymm5, ymm6, ymm7, ymm8, ymm9;
        auto inds = index(task_id, q_state_bit_num, qubits, qubits_sorted);
        if (qubits[0] == qubits_sorted[0]) {  // ...q0.q1.
          // v00 v10 v01 v11 ... v02 v12 v03 v13
          ymm4 = _mm256_loadu_pd(&real[inds[2]]);
          ymm5 = _mm256_loadu_pd(&imag[inds[2]]);
        } else {  // ...q1.q0.
          // v00 v10 v02 v12 ... v01 v11 v03 v13
          ymm4 = _mm256_loadu2_m128d(&real[inds[3]], &real[inds[2]]);
          ymm5 = _mm256_loadu2_m128d(&imag[inds[3]], &imag[inds[2]]);
        }
        ymm4 =
            _mm256_permute4x64_pd(ymm4, 0b1101'1000);  // v02 v03 v12 v13, real
        ymm5 =
            _mm256_permute4x64_pd(ymm5, 0b1101'1000);  // v02 v03 v12 v13, imag
        COMPLEX_YMM_MUL(ymm0, ymm2, ymm4, ymm5, ymm6, ymm7);
        COMPLEX_YMM_MUL(ymm1, ymm3, ymm4, ymm5, ymm8, ymm9);
        ymm4 = _mm256_hadd_pd(ymm6, ymm8);  // v02 v03 v12 v13, real
        ymm5 = _mm256_hadd_pd(ymm7, ymm9);  // v02 v03 v12 v13, imag
        ymm4 =
            _mm256_permute4x64_pd(ymm4, 0b1101'1000);  // v02 v12 v03 v13, real
        ymm5 =
            _mm256_permute4x64_pd(ymm5, 0b1101'1000);  // v02 v12 v03 v13, real
        if (qubits[0] == qubits_sorted[0]) {           // ...q0.q1.
          // v00 v10 v01 v11 ... v02 v12 v03 v13
          _mm256_storeu_pd(&real[inds[2]], ymm4);
          _mm256_storeu_pd(&imag[inds[2]], ymm5);
        } else {  // ...q1.q0.
          // v00 v10 v02 v12 ... v01 v11 v03 v13
          _mm256_storeu2_m128d(&real[inds[3]], &real[inds[2]], ymm4);
          _mm256_storeu2_m128d(&imag[inds[3]], &imag[inds[2]], ymm5);
        }
      }

    } else if (qubits_sorted[1] < q_state_bit_num - 2) {  // ...q...q..
      constexpr uint64_t batch_size = 2;
      __m256d ymm0 = _mm256_loadu2_m128d(
          &gate.mat_real_[0], &gate.mat_real_[0]);  // m0 m1 m0 m1, real
      __m256d ymm1 = _mm256_loadu2_m128d(
          &gate.mat_real_[2], &gate.mat_real_[2]);  // m2 m3 m2 m3, real
      __m256d ymm2 = _mm256_loadu2_m128d(
          &gate.mat_imag_[0], &gate.mat_imag_[0]);  // m0 m1 m0 m1, imag
      __m256d ymm3 = _mm256_loadu2_m128d(
          &gate.mat_imag_[2], &gate.mat_imag_[2]);  // m2 m3 m2 m3, imag

#pragma omp parallel for num_threads(omp_thread_num) \
    schedule(dynamic, omp_chunk_size(q_state_bit_num))
      for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
        auto inds = index(task_id, q_state_bit_num, qubits, qubits_sorted);
        __m256d ymm4 = _mm256_loadu2_m128d(
            &real[inds[3]], &real[inds[2]]);  // v02 v12 v03 v13, real
        __m256d ymm5 = _mm256_loadu2_m128d(
            &imag[inds[3]], &imag[inds[2]]);  // v02 v12 v03 v13, imag
        ymm4 =
            _mm256_permute4x64_pd(ymm4, 0b1101'1000);  // v02 v03 v12 v13, real
        ymm5 =
            _mm256_permute4x64_pd(ymm5, 0b1101'1000);  // v02 v03 v12 v13, imag
        __m256d ymm6, ymm7, ymm8, ymm9;
        COMPLEX_YMM_MUL(ymm0, ymm2, ymm4, ymm5, ymm6, ymm7);
        COMPLEX_YMM_MUL(ymm1, ymm3, ymm4, ymm5, ymm8, ymm9);
        ymm4 = _mm256_hadd_pd(ymm6, ymm8);  // v02 v03 v12 v13, real
        ymm5 = _mm256_hadd_pd(ymm7, ymm9);  // v02 v03 v12 v13, imag
        ymm4 =
            _mm256_permute4x64_pd(ymm4, 0b1101'1000);  // v02 v12 v03 v13, real
        ymm5 =
            _mm256_permute4x64_pd(ymm5, 0b1101'1000);  // v02 v12 v03 v13, real
        _mm256_storeu2_m128d(&real[inds[3]], &real[inds[2]], ymm4);
        _mm256_storeu2_m128d(&imag[inds[3]], &imag[inds[2]], ymm5);
      }
    }
  }
}
}  // namespace QuICT

#endif  // SIM_BACK_AVX_CTRL_UNITARY_GATE_TCC

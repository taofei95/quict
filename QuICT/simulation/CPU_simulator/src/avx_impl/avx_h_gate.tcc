//
// Created by Ci Lei on 2021-11-05.
//

#ifndef SIM_BACK_AVX_H_GATE_TCC
#define SIM_BACK_AVX_H_GATE_TCC

#include <cstdint>
#include <algorithm>
#include <vector>
#include <complex>
#include <utility>
#include <string>
#include <omp.h>
#include <immintrin.h>

#include "../gate.h"
#include "../utility.h"
#include "../matricks_simulator.h"

namespace QuICT {
    template<typename Precision>
    template<template<typename> class Gate>
    void MaTricksSimulator<Precision>::apply_h_gate(
            uint64_t q_state_bit_num,
            const Gate<Precision> &gate,
            Precision *real,
            Precision *imag
    ) {
        if constexpr (std::is_same_v<Precision, float>) { // float
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": "
                                     + "Not Implemented " + __func__);
        } else if constexpr (std::is_same_v<Precision, double>) { // double
            uint64_t task_num = 1ULL << (q_state_bit_num - 1);
            if (gate.targ_ == q_state_bit_num - 1) {
                constexpr uint64_t batch_size = 4;
                auto cc = gate.sqrt2_inv.real();
                __m256d ymm0 = _mm256_broadcast_sd(&cc);

#pragma omp parallel for num_threads(DEFAULT_NUM_THREADS) schedule(dynamic, omp_chunk_size(q_state_bit_num))
                for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                    auto ind_0 = index(task_id, q_state_bit_num, gate.targ_);

                    // Load
                    __m256d ymm1 = _mm256_loadu_pd(&real[ind_0[0]]);
                    __m256d ymm2 = _mm256_loadu_pd(&real[ind_0[0] + 4]);
                    __m256d ymm3 = _mm256_loadu_pd(&imag[ind_0[0]]);
                    __m256d ymm4 = _mm256_loadu_pd(&imag[ind_0[0] + 4]);
                    // Scale
                    ymm1 = _mm256_mul_pd(ymm1, ymm0);
                    ymm2 = _mm256_mul_pd(ymm2, ymm0);
                    ymm3 = _mm256_mul_pd(ymm3, ymm0);
                    ymm4 = _mm256_mul_pd(ymm4, ymm0);
                    // Horizontal arithmetic
                    __m256d ymm5 = _mm256_hadd_pd(ymm1, ymm2);
                    __m256d ymm6 = _mm256_hadd_pd(ymm3, ymm4);
                    __m256d ymm7 = _mm256_hsub_pd(ymm1, ymm2);
                    __m256d ymm8 = _mm256_hsub_pd(ymm3, ymm4);

                    ymm1 = _mm256_shuffle_pd(ymm5, ymm7, 0b0000);
                    ymm2 = _mm256_shuffle_pd(ymm5, ymm7, 0b1111);

                    ymm3 = _mm256_shuffle_pd(ymm6, ymm8, 0b0000);
                    ymm4 = _mm256_shuffle_pd(ymm6, ymm8, 0b1111);
                    // Store
                    _mm256_storeu_pd(&real[ind_0[0]], ymm1);
                    _mm256_storeu_pd(&real[ind_0[0] + 4], ymm2);
                    _mm256_storeu_pd(&imag[ind_0[0]], ymm3);
                    _mm256_storeu_pd(&imag[ind_0[0] + 4], ymm4);
                }
            } else if (gate.targ_ == q_state_bit_num - 2) {
                // After some permutations, this is the same with the previous one.
                constexpr uint64_t batch_size = 4;
                auto cc = gate.sqrt2_inv.real();
                __m256d ymm0 = _mm256_broadcast_sd(&cc);

#pragma omp parallel for num_threads(DEFAULT_NUM_THREADS) schedule(dynamic, omp_chunk_size(q_state_bit_num))
                for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                    auto ind_0 = index(task_id, q_state_bit_num, gate.targ_);


                    // Load
                    __m256d ymm1 = _mm256_loadu_pd(&real[ind_0[0]]);
                    __m256d ymm2 = _mm256_loadu_pd(&real[ind_0[0] + 4]);
                    __m256d ymm3 = _mm256_loadu_pd(&imag[ind_0[0]]);
                    __m256d ymm4 = _mm256_loadu_pd(&imag[ind_0[0] + 4]);
                    // Permute
                    ymm1 = _mm256_permute4x64_pd(ymm1, 0b1101'1000);
                    ymm2 = _mm256_permute4x64_pd(ymm2, 0b1101'1000);
                    ymm3 = _mm256_permute4x64_pd(ymm3, 0b1101'1000);
                    ymm4 = _mm256_permute4x64_pd(ymm4, 0b1101'1000);
                    // Scale
                    ymm1 = _mm256_mul_pd(ymm1, ymm0);
                    ymm2 = _mm256_mul_pd(ymm2, ymm0);
                    ymm3 = _mm256_mul_pd(ymm3, ymm0);
                    ymm4 = _mm256_mul_pd(ymm4, ymm0);
                    // Horizontal arithmetic
                    __m256d ymm5 = _mm256_hadd_pd(ymm1, ymm2);
                    __m256d ymm6 = _mm256_hadd_pd(ymm3, ymm4);
                    __m256d ymm7 = _mm256_hsub_pd(ymm1, ymm2);
                    __m256d ymm8 = _mm256_hsub_pd(ymm3, ymm4);

                    ymm1 = _mm256_shuffle_pd(ymm5, ymm7, 0b0000);
                    ymm2 = _mm256_shuffle_pd(ymm5, ymm7, 0b1111);

                    ymm3 = _mm256_shuffle_pd(ymm6, ymm8, 0b0000);
                    ymm4 = _mm256_shuffle_pd(ymm6, ymm8, 0b1111);
                    // Permute back
                    ymm1 = _mm256_permute4x64_pd(ymm1, 0b1101'1000);
                    ymm2 = _mm256_permute4x64_pd(ymm2, 0b1101'1000);
                    ymm3 = _mm256_permute4x64_pd(ymm3, 0b1101'1000);
                    ymm4 = _mm256_permute4x64_pd(ymm4, 0b1101'1000);
                    // Store
                    _mm256_storeu_pd(&real[ind_0[0]], ymm1);
                    _mm256_storeu_pd(&real[ind_0[0] + 4], ymm2);
                    _mm256_storeu_pd(&imag[ind_0[0]], ymm3);
                    _mm256_storeu_pd(&imag[ind_0[0] + 4], ymm4);
                }
            } else {
                constexpr uint64_t batch_size = 4;
                auto cc = gate.sqrt2_inv.real();
                __m256d ymm0 = _mm256_broadcast_sd(&cc);           // constant array

#pragma omp parallel for num_threads(DEFAULT_NUM_THREADS) schedule(dynamic, omp_chunk_size(q_state_bit_num))
                for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                    auto ind_0 = index(task_id, q_state_bit_num, gate.targ_);

                    // ind_0[i], ind_1[i], ind_2[i], ind_3[i] are continuous in mem

                    __m256d ymm1 = _mm256_loadu_pd(&real[ind_0[0]]);   // real_row_0
                    __m256d ymm2 = _mm256_loadu_pd(&real[ind_0[1]]);   // real_row_1
                    __m256d ymm3 = _mm256_loadu_pd(&imag[ind_0[0]]);   // imag_row_0
                    __m256d ymm4 = _mm256_loadu_pd(&imag[ind_0[1]]);   // imag_row_1

                    __m256d ymm5 = _mm256_add_pd(ymm1, ymm2);
                    __m256d ymm6 = _mm256_sub_pd(ymm1, ymm2);
                    __m256d ymm7 = _mm256_add_pd(ymm3, ymm4);
                    __m256d ymm8 = _mm256_sub_pd(ymm3, ymm4);

                    // Scale
                    ymm5 = _mm256_mul_pd(ymm0, ymm5);
                    ymm6 = _mm256_mul_pd(ymm0, ymm6);
                    ymm7 = _mm256_mul_pd(ymm0, ymm7);
                    ymm8 = _mm256_mul_pd(ymm0, ymm8);

                    _mm256_storeu_pd(&real[ind_0[0]], ymm5);
                    _mm256_storeu_pd(&real[ind_0[1]], ymm6);
                    _mm256_storeu_pd(&imag[ind_0[0]], ymm7);
                    _mm256_storeu_pd(&imag[ind_0[1]], ymm8);
                }
            }
        }
    }
}
#endif
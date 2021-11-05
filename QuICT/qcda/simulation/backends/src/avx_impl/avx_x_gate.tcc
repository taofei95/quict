//
// Created by Ci Lei on 2021-11-05.
//

#ifndef SIM_BACK_AVX_X_GATE_TCC
#define SIM_BACK_AVX_X_GATE_TCC

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

template<typename Precision>
template<template<typename> class Gate>
void MaTricksSimulator<Precision>::apply_x_gate(
        uint64_t q_state_bit_num,
        const Gate<Precision> &gate,
        Precision *real,
        Precision *imag
) {
    if constexpr(std::is_same_v<Precision, float>) {
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": "
                                 + "Not Implemented " + __func__);
    } else if constexpr(std::is_same_v<Precision, double>) {
        uint64_t task_num = 1ULL << (q_state_bit_num - 1);
        if (gate.targ_ == q_state_bit_num - 1) {
            constexpr uint64_t batch_size = 4;

#pragma omp parallel for num_threads(DEFAULT_NUM_THREADS) schedule(dynamic, omp_chunk_size(q_state_bit_num))
            for (uint64_t ind = 0; ind < (1ULL << q_state_bit_num); ind += batch_size) {
                __m256d ymm1 = _mm256_loadu_pd(&real[ind]);
                __m256d ymm2 = _mm256_loadu_pd(&imag[ind]);

                ymm1 = _mm256_permute4x64_pd(ymm1, 0b1011'0001);
                ymm2 = _mm256_permute4x64_pd(ymm2, 0b1011'0001);
                _mm256_storeu_pd(&real[ind], ymm1);
                _mm256_storeu_pd(&imag[ind], ymm2);
            }
        } else if (gate.targ_ == q_state_bit_num - 2) {
            constexpr uint64_t batch_size = 4;
#pragma omp parallel for num_threads(DEFAULT_NUM_THREADS) schedule(dynamic, omp_chunk_size(q_state_bit_num))
            for (uint64_t ind = 0; ind < (1ULL << q_state_bit_num); ind += batch_size) {
                __m256d ymm1 = _mm256_loadu_pd(&real[ind]);
                __m256d ymm2 = _mm256_loadu_pd(&imag[ind]);

                ymm1 = _mm256_permute4x64_pd(ymm1, 0b0100'1110);
                ymm2 = _mm256_permute4x64_pd(ymm2, 0b0100'1110);
                _mm256_storeu_pd(&real[ind], ymm1);
                _mm256_storeu_pd(&imag[ind], ymm2);
            }
        } else {
            constexpr uint64_t batch_size = 4;
#pragma omp parallel for num_threads(DEFAULT_NUM_THREADS) schedule(dynamic, omp_chunk_size(q_state_bit_num))
            for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                auto ind_0 = index(task_id, q_state_bit_num, gate.targ_);
                __m256d ymm1 = _mm256_loadu_pd(&real[ind_0[0]]);
                __m256d ymm2 = _mm256_loadu_pd(&real[ind_0[1]]);
                __m256d ymm3 = _mm256_loadu_pd(&imag[ind_0[0]]);
                __m256d ymm4 = _mm256_loadu_pd(&imag[ind_0[1]]);

                _mm256_storeu_pd(&real[ind_0[0]], ymm2);
                _mm256_storeu_pd(&real[ind_0[1]], ymm1);
                _mm256_storeu_pd(&imag[ind_0[0]], ymm4);
                _mm256_storeu_pd(&imag[ind_0[1]], ymm3);
            }
        }
    }
}

#endif //SIM_BACK_X_GATE_AVX_TCC

//
// Created by Ci Lei on 2021-11-05.
//

#ifndef SIM_BACK_AVX_CTRL_DIAG_GATE_TCC
#define SIM_BACK_AVX_CTRL_DIAG_GATE_TCC

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
#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"

    template<typename Precision>
    template<template<typename> class Gate>
    void MaTricksSimulator<Precision>::apply_ctrl_diag_gate(
            uint64_t q_state_bit_num,
            const Gate<Precision> &gate,
            Precision *real,
            Precision *imag
    ) {
        if constexpr(std::is_same_v<Precision, float>) {
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": "
                                     + "Not Implemented " + __func__);
        } else if constexpr(std::is_same_v<Precision, double>) {
            uarray_t<2> qubits = {gate.carg_, gate.targ_};
            uarray_t<2> qubits_sorted = {gate.carg_, gate.targ_};
            if (gate.carg_ > gate.targ_) {
                qubits_sorted[0] = gate.targ_;
                qubits_sorted[1] = gate.carg_;
            }

            uint64_t task_num = 1ULL << (q_state_bit_num - 2);
            if (qubits_sorted[1] == q_state_bit_num - 1) {
                if (qubits_sorted[0] == q_state_bit_num - 2) {

                    constexpr uint64_t blk_size = 64;
#pragma omp parallel for if(q_state_bit_num > sysconfig_.omp_threshold_) num_threads(sysconfig_.omp_num_thread_)
                    for (uint64_t blk = 0; blk < task_num; blk += blk_size) {
                        __m256d ymm0; // dr
                        __m256d ymm1; // di
                        ymm0 = _mm256_setr_pd(gate.diagonal_real_[0], gate.diagonal_real_[1],
                                              gate.diagonal_real_[0], gate.diagonal_real_[1]);
                        ymm1 = _mm256_setr_pd(gate.diagonal_imag_[0], gate.diagonal_imag_[1],
                                              gate.diagonal_imag_[0], gate.diagonal_imag_[1]);

                        constexpr uint64_t batch_size = 2;
                        Precision cache_real[blk_size * 4], cache_imag[blk_size * 4];
                        auto mem_start = index0(blk, q_state_bit_num, qubits, qubits_sorted);
                        auto mem_end = index0(blk + blk_size, q_state_bit_num, qubits, qubits_sorted);
                        // Preload
                        // std::copy is already SIMD optimized
                        std::copy(&real[mem_start], &real[mem_end], cache_real);
                        std::copy(&imag[mem_start], &imag[mem_end], cache_imag);
                        for (uint64_t task_id = 0;
                             task_id + blk < std::min(blk + blk_size, task_num);
                             task_id += batch_size) {
                            __m256d ymm2; // vr
                            __m256d ymm3; // vi
                            __m256d ymm6, ymm7; // tmp reg
//                            auto inds = index(task_id + blk, q_state_bit_num, qubits, qubits_sorted);
                            auto cur_real = &cache_real[task_id * 4];
                            auto cur_imag = &cache_imag[task_id * 4];
                            if (qubits[0] == qubits_sorted[0]) { // ...q0q1
                                // v00 v01 v02 v03 v10 v11 v12 v13
                                ymm2 = _mm256_loadu2_m128d(&cur_real[6], &cur_real[2]);
                                ymm3 = _mm256_loadu2_m128d(&cur_imag[6], &cur_imag[2]);
                            } else { // ...q1q0
                                // v00 v02 v01 v03 v10 v12 v11 v13
                                STRIDE_2_LOAD_ODD_PD(&cur_real[0], ymm2, ymm6, ymm7);
                                STRIDE_2_LOAD_ODD_PD(&cur_imag[0], ymm3, ymm6, ymm7);
                            }
                            __m256d ymm4; // res_r
                            __m256d ymm5; // res_i
                            COMPLEX_YMM_MUL(ymm0, ymm1, ymm2, ymm3, ymm4, ymm5);
                            if (qubits[0] == qubits_sorted[0]) { // ...q0q1
                                // v00 v01 v02 v03 v10 v11 v12 v13
                                _mm256_storeu2_m128d(&cur_real[6], &cur_real[2], ymm4);
                                _mm256_storeu2_m128d(&cur_imag[6], &cur_imag[2], ymm5);
                            } else { // ...q1q0
                                // v00 v02 v01 v03 v10 v12 v11 v13
                                Precision tmp[4];
                                STRIDE_2_STORE_ODD_PD(&cur_real[0], ymm4, tmp);
                                STRIDE_2_STORE_ODD_PD(&cur_imag[0], ymm5, tmp);
                            }
                        }
                        std::copy(cache_real, cache_real + blk_size * 4, &real[mem_start]);
                        std::copy(cache_imag, cache_imag + blk_size * 4, &imag[mem_start]);
                    }
                } else if (qubits_sorted[0] < q_state_bit_num - 2) {
                    if (qubits_sorted[0] == qubits[0]) { // ...q0.q1
                        // v00 v01 v10 v11 . v02 v03 v12 v13

                        constexpr uint64_t blk_size = 64;

#pragma omp parallel for if(q_state_bit_num > sysconfig_.omp_threshold_) num_threads(sysconfig_.omp_num_thread_)
                        for (uint64_t blk = 0; blk < task_num; blk += blk_size) {
                            Precision c_arr_real[4] =
                                    {gate.diagonal_real_[0], gate.diagonal_real_[1],
                                     gate.diagonal_real_[0], gate.diagonal_real_[1]};
                            Precision c_arr_imag[4] =
                                    {gate.diagonal_imag_[0], gate.diagonal_imag_[1],
                                     gate.diagonal_imag_[0], gate.diagonal_imag_[1]};
                            __m256d ymm0 = _mm256_loadu_pd(c_arr_real);
                            __m256d ymm1 = _mm256_loadu_pd(c_arr_imag);

                            constexpr uint64_t batch_size = 2;
                            Precision cache_real[blk_size * 4], cache_imag[blk_size * 4];
                            // Preload
                            for (uint64_t task_id = 0;
                                 task_id + blk < std::min(blk + blk_size, task_num);
                                 task_id += batch_size) {
                                auto inds = index(task_id + blk, q_state_bit_num, qubits, qubits_sorted);
//                                std::copy(&real[inds[0]], &real[inds[0] + 4], &cache_real[task_id * 4]);
                                std::copy(&real[inds[2]], &real[inds[2] + 4], &cache_real[(task_id + 1) * 4]);
//                                std::copy(&imag[inds[0]], &imag[inds[0] + 4], &cache_imag[task_id * 4]);
                                std::copy(&imag[inds[2]], &imag[inds[2] + 4], &cache_imag[(task_id + 1) * 4]);
                            }
                            for (uint64_t task_id = 0;
                                 task_id + blk < std::min(blk + blk_size, task_num);
                                 task_id += batch_size) {
//                                auto ind0 = index0(task_id + blk, q_state_bit_num, qubits, qubits_sorted);
                                auto cur_real = &cache_real[(task_id + 1) * 4];
                                auto cur_imag = &cache_imag[(task_id + 1) * 4];
                                __m256d ymm2 = _mm256_loadu_pd(cur_real);  // vr
                                __m256d ymm3 = _mm256_loadu_pd(cur_imag);  // vi
                                __m256d ymm4;  // res_r
                                __m256d ymm5;  // res_i
                                COMPLEX_YMM_MUL(ymm0, ymm1, ymm2, ymm3, ymm4, ymm5);
                                _mm256_storeu_pd(cur_real, ymm4);
                                _mm256_storeu_pd(cur_imag, ymm5);
                            }
                            for (uint64_t task_id = 0;
                                 task_id + blk < std::min(blk + blk_size, task_num);
                                 task_id += batch_size) {
                                auto inds = index(task_id + blk, q_state_bit_num, qubits, qubits_sorted);
//                                std::copy(&cache_real[task_id * 4],
//                                          &cache_real[(task_id + 1) * 4],
//                                          &real[inds[0]]);
                                std::copy(&cache_real[(task_id + 1) * 4],
                                          &cache_real[(task_id + 2) * 4],
                                          &real[inds[2]]);
//                                std::copy(&cache_imag[task_id * 4],
//                                          &cache_imag[(task_id + 1) * 4],
//                                          &imag[inds[0]]);
                                std::copy(&cache_imag[(task_id + 1) * 4],
                                          &cache_imag[(task_id + 2) * 4],
                                          &imag[inds[2]]);
                            }
                        }
                    } else { // ...q1.q0
                        // v00 v02 v10 v12 . v01 v03 v11 v13

                        constexpr uint64_t blk_size = 64;

#pragma omp parallel for if(q_state_bit_num > sysconfig_.omp_threshold_) num_threads(sysconfig_.omp_num_thread_)
                        for (uint64_t blk = 0; blk < task_num; blk += blk_size) {
                            __m256d ymm0 = _mm256_setr_pd(gate.diagonal_real_[0], gate.diagonal_real_[1],
                                                          gate.diagonal_real_[0], gate.diagonal_real_[1]);
                            __m256d ymm1 = _mm256_setr_pd(gate.diagonal_imag_[0], gate.diagonal_imag_[1],
                                                          gate.diagonal_imag_[0], gate.diagonal_imag_[1]);

                            constexpr uint64_t batch_size = 2;
                            Precision cache_real[blk_size * 4], cache_imag[blk_size * 4];
                            // Preload
                            for (uint64_t task_id = 0;
                                 task_id + blk < std::min(blk + blk_size, task_num);
                                 task_id += batch_size) {
                                auto inds = index(task_id + blk, q_state_bit_num, qubits, qubits_sorted);
                                std::copy(&real[inds[0]], &real[inds[0] + 4], &cache_real[task_id * 4]);
                                std::copy(&real[inds[1]], &real[inds[1] + 4], &cache_real[(task_id + 1) * 4]);
                                std::copy(&imag[inds[0]], &imag[inds[0] + 4], &cache_imag[task_id * 4]);
                                std::copy(&imag[inds[1]], &imag[inds[1] + 4], &cache_imag[(task_id + 1) * 4]);
                            }
                            for (uint64_t task_id = 0;
                                 task_id + blk < std::min(blk + blk_size, task_num);
                                 task_id += batch_size) {
//                                auto inds = index(task_id + blk, q_state_bit_num, qubits, qubits_sorted);
                                auto cur_real = &cache_real[task_id * 4];
                                auto cur_imag = &cache_imag[task_id * 4];
                                __m256d ymm2 = _mm256_loadu_pd(cur_real); // v00 v02 v10 v12, real
                                __m256d ymm3 = _mm256_loadu_pd(&cur_real[4]); // v01 v03 v11 v13, real
                                __m256d ymm4 = _mm256_loadu_pd(cur_imag); // v00 v02 v10 v12, imag
                                __m256d ymm5 = _mm256_loadu_pd(&cur_imag[4]); // v01 v03 v11 v13, imag

                                __m256d ymm6 = _mm256_shuffle_pd(ymm2, ymm3, 0b0000); // v00 v01 v10 v11, real
                                __m256d ymm7 = _mm256_shuffle_pd(ymm2, ymm3, 0b1111); // v02 v03 v12 v13, real
                                __m256d ymm8 = _mm256_shuffle_pd(ymm4, ymm5, 0b0000); // v00 v01 v10 v11, imag
                                __m256d ymm9 = _mm256_shuffle_pd(ymm4, ymm5, 0b1111); // v02 v03 v12 v13, imag

                                __m256d ymm10, ymm11; // res_r, res_i
                                COMPLEX_YMM_MUL(ymm0, ymm1, ymm7, ymm9, ymm10, ymm11);
                                ymm2 = _mm256_shuffle_pd(ymm6, ymm10, 0b0000);
                                ymm3 = _mm256_shuffle_pd(ymm6, ymm10, 0b1111);
                                ymm4 = _mm256_shuffle_pd(ymm8, ymm11, 0b0000);
                                ymm5 = _mm256_shuffle_pd(ymm8, ymm11, 0b1111);

                                _mm256_storeu_pd(cur_real, ymm2);
                                _mm256_storeu_pd(&cur_real[4], ymm3);
                                _mm256_storeu_pd(cur_imag, ymm4);
                                _mm256_storeu_pd(&cur_imag[4], ymm5);
                            }
                            for (uint64_t task_id = 0;
                                 task_id + blk < std::min(blk + blk_size, task_num);
                                 task_id += batch_size) {
                                auto inds = index(task_id + blk, q_state_bit_num, qubits, qubits_sorted);
                                std::copy(&cache_real[task_id * 4],
                                          &cache_real[(task_id + 1) * 4],
                                          &real[inds[0]]);
                                std::copy(&cache_real[(task_id + 1) * 4],
                                          &cache_real[(task_id + 2) * 4],
                                          &real[inds[1]]);
                                std::copy(&cache_imag[task_id * 4],
                                          &cache_imag[(task_id + 1) * 4],
                                          &imag[inds[0]]);
                                std::copy(&cache_imag[(task_id + 1) * 4],
                                          &cache_imag[(task_id + 2) * 4],
                                          &imag[inds[1]]);
                            }
                        }
                    }
                }
            } else if (qubits_sorted[1] == q_state_bit_num - 2) {
                // ...q.q.
                if (qubits[0] == qubits_sorted[0]) { // ...q0.q1.
                    // v00 v10 v01 v11 ... v02 v12 v03 v13

                    constexpr uint64_t blk_size = 64;

#pragma omp parallel for if(q_state_bit_num > sysconfig_.omp_threshold_) num_threads(sysconfig_.omp_num_thread_)
                    for (uint64_t blk = 0; blk < task_num; blk += blk_size) {
                        Precision c_arr_real[4] =
                                {gate.diagonal_real_[0], gate.diagonal_real_[0],
                                 gate.diagonal_real_[1], gate.diagonal_real_[1]};
                        Precision c_arr_imag[4] =
                                {gate.diagonal_imag_[0], gate.diagonal_imag_[0],
                                 gate.diagonal_imag_[1], gate.diagonal_imag_[1]};
                        __m256d ymm0 = _mm256_loadu_pd(c_arr_real); // dr
                        __m256d ymm1 = _mm256_loadu_pd(c_arr_imag); // di

                        constexpr uint64_t batch_size = 2;
                        Precision cache_real[blk_size * 4], cache_imag[blk_size * 4];
                        // Preload
                        for (uint64_t task_id = 0;
                             task_id + blk < std::min(blk + blk_size, task_num);
                             task_id += batch_size) {
                            auto inds = index(task_id + blk, q_state_bit_num, qubits, qubits_sorted);
                            std::copy(&real[inds[0]], &real[inds[0] + 4], &cache_real[task_id * 4]);
                            std::copy(&real[inds[2]], &real[inds[2] + 4], &cache_real[(task_id + 1) * 4]);
                            std::copy(&imag[inds[0]], &imag[inds[0] + 4], &cache_imag[task_id * 4]);
                            std::copy(&imag[inds[2]], &imag[inds[2] + 4], &cache_imag[(task_id + 1) * 4]);
                        }
                        for (uint64_t task_id = 0;
                             task_id + blk < std::min(blk + blk_size, task_num);
                             task_id += batch_size) {
//                            auto inds = index(task_id + blk, q_state_bit_num, qubits, qubits_sorted);
                            auto cur_real = &cache_real[(task_id + 1) * 4];
                            auto cur_imag = &cache_imag[(task_id + 1) * 4];
                            __m256d ymm2 = _mm256_loadu_pd(cur_real); // vr
                            __m256d ymm3 = _mm256_loadu_pd(cur_imag);  // vi
                            __m256d ymm4, ymm5;
                            COMPLEX_YMM_MUL(ymm0, ymm1, ymm2, ymm3, ymm4, ymm5);
                            _mm256_storeu_pd(cur_real, ymm4);
                            _mm256_storeu_pd(cur_imag, ymm5);
                        }
                        for (uint64_t task_id = 0;
                             task_id + blk < std::min(blk + blk_size, task_num);
                             task_id += batch_size) {
                            auto inds = index(task_id + blk, q_state_bit_num, qubits, qubits_sorted);
                            std::copy(&cache_real[task_id * 4],
                                      &cache_real[(task_id + 1) * 4],
                                      &real[inds[0]]);
                            std::copy(&cache_real[(task_id + 1) * 4],
                                      &cache_real[(task_id + 2) * 4],
                                      &real[inds[2]]);
                            std::copy(&cache_imag[task_id * 4],
                                      &cache_imag[(task_id + 1) * 4],
                                      &imag[inds[0]]);
                            std::copy(&cache_imag[(task_id + 1) * 4],
                                      &cache_imag[(task_id + 2) * 4],
                                      &imag[inds[2]]);
                        }
                    }
                } else { // ...q1.q0.
                    // v00 v10 v02 v12 ... v01 v11 v03 v13

                    constexpr uint64_t blk_size = 64;

#pragma omp parallel for if(q_state_bit_num > sysconfig_.omp_threshold_) num_threads(sysconfig_.omp_num_thread_)
                    for (uint64_t blk = 0; blk < task_num; blk += blk_size) {
                        __m256d ymm0 = _mm256_setr_pd(gate.diagonal_real_[0], gate.diagonal_real_[0],
                                                      gate.diagonal_real_[1], gate.diagonal_real_[1]); // dr
                        __m256d ymm1 = _mm256_setr_pd(gate.diagonal_imag_[0], gate.diagonal_imag_[0],
                                                      gate.diagonal_imag_[1], gate.diagonal_imag_[1]); //di

                        constexpr uint64_t batch_size = 2;
                        Precision cache_real[blk_size * 4], cache_imag[blk_size * 4];
                        // Preload
                        for (uint64_t task_id = 0;
                             task_id + blk < std::min(blk + blk_size, task_num);
                             task_id += batch_size) {
                            auto inds = index(task_id + blk, q_state_bit_num, qubits, qubits_sorted);
                            std::copy(&real[inds[0]], &real[inds[0] + 4], &cache_real[task_id * 4]);
                            std::copy(&real[inds[1]], &real[inds[1] + 4], &cache_real[(task_id + 1) * 4]);
                            std::copy(&imag[inds[0]], &imag[inds[0] + 4], &cache_imag[task_id * 4]);
                            std::copy(&imag[inds[1]], &imag[inds[1] + 4], &cache_imag[(task_id + 1) * 4]);
                        }
                        for (uint64_t task_id = 0;
                             task_id + blk < std::min(blk + blk_size, task_num);
                             task_id += batch_size) {
//                            auto inds = index(task_id + blk, q_state_bit_num, qubits, qubits_sorted);
                            auto cur_real = &cache_real[task_id * 4];
                            auto cur_imag = &cache_imag[task_id * 4];
                            __m256d ymm2 = _mm256_loadu2_m128d(&cur_real[6], &cur_real[2]); // vr
                            __m256d ymm3 = _mm256_loadu2_m128d(&cur_imag[6], &cur_imag[2]); // vi
                            __m256d ymm4; // res_r
                            __m256d ymm5; // res_i
                            COMPLEX_YMM_MUL(ymm0, ymm1, ymm2, ymm3, ymm4, ymm5);
                            _mm256_storeu2_m128d(&cur_real[6], &cur_real[2], ymm4);
                            _mm256_storeu2_m128d(&cur_imag[6], &cur_imag[2], ymm5);
                        }
                        for (uint64_t task_id = 0;
                             task_id + blk < std::min(blk + blk_size, task_num);
                             task_id += batch_size) {
                            auto inds = index(task_id + blk, q_state_bit_num, qubits, qubits_sorted);
                            std::copy(&cache_real[task_id * 4],
                                      &cache_real[(task_id + 1) * 4],
                                      &real[inds[0]]);
                            std::copy(&cache_real[(task_id + 1) * 4],
                                      &cache_real[(task_id + 2) * 4],
                                      &real[inds[1]]);
                            std::copy(&cache_imag[task_id * 4],
                                      &cache_imag[(task_id + 1) * 4],
                                      &imag[inds[0]]);
                            std::copy(&cache_imag[(task_id + 1) * 4],
                                      &cache_imag[(task_id + 2) * 4],
                                      &imag[inds[1]]);
                        }
                    }
                }
            } else if (qubits_sorted[1] < q_state_bit_num - 2) { // ...q...q..
                // Easiest branch :)
                __m256d ymm0 = _mm256_broadcast_sd(&gate.diagonal_real_[0]);
                __m256d ymm1 = _mm256_broadcast_sd(&gate.diagonal_real_[1]);
                __m256d ymm2 = _mm256_broadcast_sd(&gate.diagonal_imag_[0]);
                __m256d ymm3 = _mm256_broadcast_sd(&gate.diagonal_imag_[1]);
                constexpr uint64_t batch_size = 4;

#pragma omp parallel for if(q_state_bit_num > sysconfig_.omp_threshold_) num_threads(sysconfig_.omp_num_thread_)
                for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                    auto inds = index(task_id, q_state_bit_num, qubits, qubits_sorted);
                    __m256d ymm4 = _mm256_loadu_pd(&real[inds[2]]);
                    __m256d ymm5 = _mm256_loadu_pd(&real[inds[3]]);
                    __m256d ymm6 = _mm256_loadu_pd(&imag[inds[2]]);
                    __m256d ymm7 = _mm256_loadu_pd(&imag[inds[3]]);

                    __m256d ymm8, ymm9, ymm10, ymm11;
                    COMPLEX_YMM_MUL(ymm0, ymm2, ymm4, ymm6, ymm8, ymm9);
                    COMPLEX_YMM_MUL(ymm1, ymm3, ymm5, ymm7, ymm10, ymm11);
                    _mm256_storeu_pd(&real[inds[2]], ymm8);
                    _mm256_storeu_pd(&real[inds[3]], ymm10);
                    _mm256_storeu_pd(&imag[inds[2]], ymm9);
                    _mm256_storeu_pd(&imag[inds[3]], ymm11);
                }
            }
        }
    }

#pragma clang diagnostic pop


//    template<typename Precision>
//    template<template<typename> class Gate>
//    void MaTricksSimulator<Precision>::apply_ctrl_diag_gate(
//            uint64_t q_state_bit_num,
//            const Gate<Precision> &gate,
//            Precision *real,
//            Precision *imag
//    ) {
//        if constexpr(std::is_same_v<Precision, float>) {
//            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": "
//                                     + "Not Implemented " + __func__);
//        } else if constexpr(std::is_same_v<Precision, double>) {
//            uarray_t<2> qubits = {gate.carg_, gate.targ_};
//            uarray_t<2> qubits_sorted = {gate.carg_, gate.targ_};
//            if (gate.carg_ > gate.targ_) {
//                qubits_sorted[0] = gate.targ_;
//                qubits_sorted[1] = gate.carg_;
//            }
//
//            uint64_t task_num = 1ULL << (q_state_bit_num - 2);
//            if (qubits_sorted[1] == q_state_bit_num - 1) {
//                if (qubits_sorted[0] == q_state_bit_num - 2) {
//                    __m256d ymm0; // dr
//                    __m256d ymm1; // di
//                    ymm0 = _mm256_setr_pd(gate.diagonal_real_[0], gate.diagonal_real_[1],
//                                          gate.diagonal_real_[0], gate.diagonal_real_[1]);
//                    ymm1 = _mm256_setr_pd(gate.diagonal_imag_[0], gate.diagonal_imag_[1],
//                                          gate.diagonal_imag_[0], gate.diagonal_imag_[1]);
//                    constexpr uint64_t batch_size = 2;
//
//#pragma omp parallel for num_threads(DEFAULT_NUM_THREADS) schedule(dynamic, omp_chunk_size(q_state_bit_num))
//                    for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
//                        __m256d ymm2; // vr
//                        __m256d ymm3; // vi
//                        __m256d ymm6, ymm7; // tmp reg
//                        auto inds = index(task_id, q_state_bit_num, qubits, qubits_sorted);
//                        if (qubits[0] == qubits_sorted[0]) { // ...q0q1
//                            // v00 v01 v02 v03 v10 v11 v12 v13
//                            ymm2 = _mm256_loadu2_m128d(&real[inds[2] + 4], &real[inds[2]]);
//                            ymm3 = _mm256_loadu2_m128d(&imag[inds[2] + 4], &imag[inds[2]]);
//                        } else { // ...q1q0
//                            // v00 v02 v01 v03 v10 v12 v11 v13
//                            STRIDE_2_LOAD_ODD_PD(&real[inds[0]], ymm2, ymm6, ymm7);
//                            STRIDE_2_LOAD_ODD_PD(&imag[inds[0]], ymm3, ymm6, ymm7);
//                        }
//                        __m256d ymm4; // res_r
//                        __m256d ymm5; // res_i
//                        COMPLEX_YMM_MUL(ymm0, ymm1, ymm2, ymm3, ymm4, ymm5);
//                        if (qubits[0] == qubits_sorted[0]) { // ...q0q1
//                            // v00 v01 v02 v03 v10 v11 v12 v13
//                            _mm256_storeu2_m128d(&real[inds[2] + 4], &real[inds[2]], ymm4);
//                            _mm256_storeu2_m128d(&imag[inds[2] + 4], &imag[inds[2]], ymm5);
//                        } else { // ...q1q0
//                            // v00 v02 v01 v03 v10 v12 v11 v13
//                            Precision tmp[4];
//                            STRIDE_2_STORE_ODD_PD(&real[inds[0]], ymm4, tmp);
//                            STRIDE_2_STORE_ODD_PD(&imag[inds[0]], ymm5, tmp);
//                        }
//                    }
//                } else if (qubits_sorted[0] < q_state_bit_num - 2) {
//                    if (qubits_sorted[0] == qubits[0]) { // ...q0.q1
//                        // v00 v01 v10 v11 . v02 v03 v12 v13
//                        constexpr uint64_t batch_size = 2;
//                        Precision c_arr_real[4] =
//                                {gate.diagonal_real_[0], gate.diagonal_real_[1],
//                                 gate.diagonal_real_[0], gate.diagonal_real_[1]};
//                        Precision c_arr_imag[4] =
//                                {gate.diagonal_imag_[0], gate.diagonal_imag_[1],
//                                 gate.diagonal_imag_[0], gate.diagonal_imag_[1]};
//                        __m256d ymm0 = _mm256_loadu_pd(c_arr_real);
//                        __m256d ymm1 = _mm256_loadu_pd(c_arr_imag);
//
//#pragma omp parallel for num_threads(DEFAULT_NUM_THREADS) schedule(dynamic, omp_chunk_size(q_state_bit_num))
//                        for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
//                            auto inds = index(task_id, q_state_bit_num, qubits, qubits_sorted);
//                            __m256d ymm2 = _mm256_loadu_pd(&real[inds[2]]);  // vr
//                            __m256d ymm3 = _mm256_loadu_pd(&imag[inds[2]]);  // vi
//                            __m256d ymm4;  // res_r
//                            __m256d ymm5;  // res_i
//                            COMPLEX_YMM_MUL(ymm0, ymm1, ymm2, ymm3, ymm4, ymm5);
//                            _mm256_storeu_pd(&real[inds[2]], ymm4);
//                            _mm256_storeu_pd(&imag[inds[2]], ymm5);
//                        }
//                    } else { // ...q1.q0
//                        // v00 v02 v10 v12 . v01 v03 v11 v13
//                        constexpr uint64_t batch_size = 2;
//                        __m256d ymm0 = _mm256_setr_pd(gate.diagonal_real_[0], gate.diagonal_real_[1],
//                                                      gate.diagonal_real_[0], gate.diagonal_real_[1]);
//                        __m256d ymm1 = _mm256_setr_pd(gate.diagonal_imag_[0], gate.diagonal_imag_[1],
//                                                      gate.diagonal_imag_[0], gate.diagonal_imag_[1]);
//
//#pragma omp parallel for num_threads(DEFAULT_NUM_THREADS) schedule(dynamic, omp_chunk_size(q_state_bit_num))
//                        for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
//                            auto inds = index(task_id, q_state_bit_num, qubits, qubits_sorted);
//                            __m256d ymm2 = _mm256_loadu_pd(&real[inds[0]]); // v00 v02 v10 v12, real
//                            __m256d ymm3 = _mm256_loadu_pd(&real[inds[1]]); // v01 v03 v11 v13, real
//                            __m256d ymm4 = _mm256_loadu_pd(&imag[inds[0]]); // v00 v02 v10 v12, imag
//                            __m256d ymm5 = _mm256_loadu_pd(&imag[inds[1]]); // v01 v03 v11 v13, imag
//
//                            __m256d ymm6 = _mm256_shuffle_pd(ymm2, ymm3, 0b0000); // v00 v01 v10 v11, real
//                            __m256d ymm7 = _mm256_shuffle_pd(ymm2, ymm3, 0b1111); // v02 v03 v12 v13, real
//                            __m256d ymm8 = _mm256_shuffle_pd(ymm4, ymm5, 0b0000); // v00 v01 v10 v11, imag
//                            __m256d ymm9 = _mm256_shuffle_pd(ymm4, ymm5, 0b1111); // v02 v03 v12 v13, imag
//
//                            __m256d ymm10, ymm11; // res_r, res_i
//                            COMPLEX_YMM_MUL(ymm0, ymm1, ymm7, ymm9, ymm10, ymm11);
//                            ymm2 = _mm256_shuffle_pd(ymm6, ymm10, 0b0000);
//                            ymm3 = _mm256_shuffle_pd(ymm6, ymm10, 0b1111);
//                            ymm4 = _mm256_shuffle_pd(ymm8, ymm11, 0b0000);
//                            ymm5 = _mm256_shuffle_pd(ymm8, ymm11, 0b1111);
//
//                            _mm256_storeu_pd(&real[inds[0]], ymm2);
//                            _mm256_storeu_pd(&real[inds[1]], ymm3);
//                            _mm256_storeu_pd(&imag[inds[0]], ymm4);
//                            _mm256_storeu_pd(&imag[inds[1]], ymm5);
//                        }
//                    }
//                }
//            } else if (qubits_sorted[1] == q_state_bit_num - 2) {
//                // ...q.q.
//                // Test Passed 2021-09-11
//                if (qubits[0] == qubits_sorted[0]) { // ...q0.q1.
//                    // v00 v10 v01 v11 ... v02 v12 v03 v13
//                    constexpr uint64_t batch_size = 2;
//                    Precision c_arr_real[4] =
//                            {gate.diagonal_real_[0], gate.diagonal_real_[0],
//                             gate.diagonal_real_[1], gate.diagonal_real_[1]};
//                    Precision c_arr_imag[4] =
//                            {gate.diagonal_imag_[0], gate.diagonal_imag_[0],
//                             gate.diagonal_imag_[1], gate.diagonal_imag_[1]};
//                    __m256d ymm0 = _mm256_loadu_pd(c_arr_real); // dr
//                    __m256d ymm1 = _mm256_loadu_pd(c_arr_imag); // di
//
//#pragma omp parallel for num_threads(DEFAULT_NUM_THREADS) schedule(dynamic, omp_chunk_size(q_state_bit_num))
//                    for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
//                        auto inds = index(task_id, q_state_bit_num, qubits, qubits_sorted);
//                        __m256d ymm2 = _mm256_loadu_pd(&real[inds[2]]); // vr
//                        __m256d ymm3 = _mm256_loadu_pd(&imag[inds[2]]);  // vi
//                        __m256d ymm4, ymm5;
//                        COMPLEX_YMM_MUL(ymm0, ymm1, ymm2, ymm3, ymm4, ymm5);
//                        _mm256_storeu_pd(&real[inds[2]], ymm4);
//                        _mm256_storeu_pd(&imag[inds[2]], ymm5);
//                    }
//                } else { // ...q1.q0.
//                    // v00 v10 v02 v12 ... v01 v11 v03 v13
//                    constexpr uint64_t batch_size = 2;
//                    __m256d ymm0 = _mm256_setr_pd(gate.diagonal_real_[0], gate.diagonal_real_[0],
//                                                  gate.diagonal_real_[1], gate.diagonal_real_[1]); // dr
//                    __m256d ymm1 = _mm256_setr_pd(gate.diagonal_imag_[0], gate.diagonal_imag_[0],
//                                                  gate.diagonal_imag_[1], gate.diagonal_imag_[1]); //di
//
//#pragma omp parallel for num_threads(DEFAULT_NUM_THREADS) schedule(dynamic, omp_chunk_size(q_state_bit_num))
//                    for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
//                        auto inds = index(task_id, q_state_bit_num, qubits, qubits_sorted);
//                        __m256d ymm2 = _mm256_loadu2_m128d(&real[inds[1] + 2], &real[inds[0] + 2]); // vr
//                        __m256d ymm3 = _mm256_loadu2_m128d(&imag[inds[1] + 2], &imag[inds[0] + 2]); // vi
//                        __m256d ymm4; // res_r
//                        __m256d ymm5; // res_i
//                        COMPLEX_YMM_MUL(ymm0, ymm1, ymm2, ymm3, ymm4, ymm5);
//                        _mm256_storeu2_m128d(&real[inds[1] + 2], &real[inds[0] + 2], ymm4);
//                        _mm256_storeu2_m128d(&imag[inds[1] + 2], &imag[inds[0] + 2], ymm5);
//                    }
//                }
//            } else if (qubits_sorted[1] < q_state_bit_num - 2) { // ...q...q..
//                // Easiest branch :)
//                __m256d ymm0 = _mm256_broadcast_sd(&gate.diagonal_real_[0]);
//                __m256d ymm1 = _mm256_broadcast_sd(&gate.diagonal_real_[1]);
//                __m256d ymm2 = _mm256_broadcast_sd(&gate.diagonal_imag_[0]);
//                __m256d ymm3 = _mm256_broadcast_sd(&gate.diagonal_imag_[1]);
//                constexpr uint64_t batch_size = 4;
//
//#pragma omp parallel for num_threads(DEFAULT_NUM_THREADS) schedule(dynamic, omp_chunk_size(q_state_bit_num))
//                for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
//                    auto inds = index(task_id, q_state_bit_num, qubits, qubits_sorted);
//                    __m256d ymm4 = _mm256_loadu_pd(&real[inds[2]]);
//                    __m256d ymm5 = _mm256_loadu_pd(&real[inds[3]]);
//                    __m256d ymm6 = _mm256_loadu_pd(&imag[inds[2]]);
//                    __m256d ymm7 = _mm256_loadu_pd(&imag[inds[3]]);
//
//                    __m256d ymm8, ymm9, ymm10, ymm11;
//                    COMPLEX_YMM_MUL(ymm0, ymm2, ymm4, ymm6, ymm8, ymm9);
//                    COMPLEX_YMM_MUL(ymm1, ymm3, ymm5, ymm7, ymm10, ymm11);
//                    _mm256_storeu_pd(&real[inds[2]], ymm8);
//                    _mm256_storeu_pd(&real[inds[3]], ymm10);
//                    _mm256_storeu_pd(&imag[inds[2]], ymm9);
//                    _mm256_storeu_pd(&imag[inds[3]], ymm11);
//                }
//            }
//        }
//    }
}

#endif //SIM_BACK_AVX_CTRL_DIAG_GATE_TCC

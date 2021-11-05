//
// Created by Ci Lei on 2021-11-05.
//

#ifndef SIM_BACK_AVX_UNITARY_N_GATE_TCC
#define SIM_BACK_AVX_UNITARY_N_GATE_TCC

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
    template<uint64_t N, template<uint64_t, typename> class Gate>
    void MaTricksSimulator<Precision>::apply_unitary_n_gate(
            uint64_t q_state_bit_num,
            const Gate<N, Precision> &gate,
            Precision *real,
            Precision *imag
    ) {
#define TWICE(x) (x), (x)
        if constexpr(std::is_same_v<Precision, float>) {
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": "
                                     + "Not Implemented " + __func__);
        } else if constexpr(std::is_same_v<Precision, double>) {
            if constexpr(N == 1) {
                uint64_t task_num = 1ULL << (q_state_bit_num - 1);
                if (gate.targ_ == q_state_bit_num - 1) {
                    // op0 := {a00, a01, a00, a01}
                    // op1 := {a10, a11, a10, a11}
                    __m256d op_re[2], op_im[2];
                    for (int i = 0; i < 2; i++) {
                        op_re[i] = _mm256_setr_pd(gate.mat_real_[i << 1], gate.mat_real_[(i << 1) | 1],
                                                  gate.mat_real_[i << 1], gate.mat_real_[(i << 1) | 1]);
                        op_im[i] = _mm256_setr_pd(gate.mat_imag_[i << 1], gate.mat_imag_[(i << 1) | 1],
                                                  gate.mat_imag_[i << 1], gate.mat_imag_[(i << 1) | 1]);
                    }

                    constexpr uint64_t batch_size = 4;

                    {
#pragma omp parallel for num_threads(DEFAULT_NUM_THREADS) schedule(dynamic, omp_chunk_size(q_state_bit_num))
                        for (int i = 0; i < (1 << q_state_bit_num); i += batch_size) {
                            __m256d re = _mm256_loadu_pd(&real[i]);
                            __m256d im = _mm256_loadu_pd(&imag[i]);
                            __m256d res_re[2], res_im[2];
                            COMPLEX_YMM_MUL(re, im, op_re[0], op_im[0], res_re[0], res_im[0]);
                            COMPLEX_YMM_MUL(re, im, op_re[1], op_im[1], res_re[1], res_im[1]);

                            re = _mm256_hadd_pd(res_re[0], res_re[1]);
                            im = _mm256_hadd_pd(res_im[0], res_im[1]);
                            _mm256_storeu_pd(&real[i], re);
                            _mm256_storeu_pd(&imag[i], im);
                        }
                    }
                } else if (gate.targ_ == q_state_bit_num - 2) {
                    __m256d op_re[2], op_im[2];
                    for (int i = 0; i < 2; i++) {
                        op_re[i] = _mm256_setr_pd(gate.mat_real_[i << 1], gate.mat_real_[i << 1],
                                                  gate.mat_real_[(i << 1) | 1], gate.mat_real_[(i << 1) | 1]);
                        op_im[i] = _mm256_setr_pd(gate.mat_imag_[i << 1], gate.mat_imag_[i << 1],
                                                  gate.mat_imag_[(i << 1) | 1], gate.mat_imag_[(i << 1) | 1]);
                    }

                    constexpr uint64_t batch_size = 4;

                    {
#pragma omp parallel for num_threads(DEFAULT_NUM_THREADS) schedule(dynamic, omp_chunk_size(q_state_bit_num))
                        for (int i = 0; i < (1 << q_state_bit_num); i += batch_size) {
                            __m256d re = _mm256_loadu_pd(&real[i]);
                            __m256d im = _mm256_loadu_pd(&imag[i]);
                            __m256d res_re[2], res_im[2];

                            COMPLEX_YMM_MUL(re, im, op_re[0], op_im[0], res_re[0], res_im[0]);
                            COMPLEX_YMM_MUL(re, im, op_re[1], op_im[1], res_re[1], res_im[1]);

                            res_re[0] = _mm256_permute4x64_pd(res_re[0], 0b1101'1000);
                            res_re[1] = _mm256_permute4x64_pd(res_re[1], 0b1101'1000);
                            res_im[0] = _mm256_permute4x64_pd(res_im[0], 0b1101'1000);
                            res_im[1] = _mm256_permute4x64_pd(res_im[1], 0b1101'1000);

                            re = _mm256_hadd_pd(res_re[0], res_re[1]);
                            im = _mm256_hadd_pd(res_im[0], res_im[1]);
                            re = _mm256_permute4x64_pd(re, 0b1101'1000);
                            im = _mm256_permute4x64_pd(im, 0b1101'1000);
                            _mm256_storeu_pd(&real[i], re);
                            _mm256_storeu_pd(&imag[i], im);
                        }
                    }
                } else {
                    constexpr uint64_t batch_size = 4;
                    __m256d op_re[4], op_im[4];
                    for (int i = 0; i < 4; i++) {
                        op_re[i] = _mm256_set1_pd(gate.mat_real_[i]);
                        op_im[i] = _mm256_set1_pd(gate.mat_imag_[i]);
                    }

                    {
#pragma omp parallel for num_threads(DEFAULT_NUM_THREADS) schedule(dynamic, omp_chunk_size(q_state_bit_num))
                        for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                            auto ind = index(task_id, q_state_bit_num, gate.targ_);
                            __m256d i0_re = _mm256_loadu_pd(&real[ind[0]]);
                            __m256d i0_im = _mm256_loadu_pd(&imag[ind[0]]);
                            __m256d i1_re = _mm256_loadu_pd(&real[ind[1]]);
                            __m256d i1_im = _mm256_loadu_pd(&imag[ind[1]]);

                            __m256d r0_re, r0_im, r1_re, r1_im;
                            COMPLEX_YMM_MUL(i0_re, i0_im, op_re[0], op_im[0], r0_re, r0_im);
                            COMPLEX_YMM_MUL(i1_re, i1_im, op_re[1], op_im[1], r1_re, r1_im);
                            r0_re = _mm256_add_pd(r0_re, r1_re);
                            r0_im = _mm256_add_pd(r0_im, r1_im);

                            _mm256_storeu_pd(&real[ind[0]], r0_re);
                            _mm256_storeu_pd(&imag[ind[0]], r0_im);

                            COMPLEX_YMM_MUL(i0_re, i0_im, op_re[2], op_im[2], r0_re, r0_im);
                            COMPLEX_YMM_MUL(i1_re, i1_im, op_re[3], op_im[3], r1_re, r1_im);
                            r0_re = _mm256_add_pd(r0_re, r1_re);
                            r0_im = _mm256_add_pd(r0_im, r1_im);

                            _mm256_storeu_pd(&real[ind[1]], r0_re);
                            _mm256_storeu_pd(&imag[ind[1]], r0_im);
                        }
                    }
                }
            } else if constexpr (N == 2) {
                uarray_t<2> qubits = {gate.affect_args_[0], gate.affect_args_[1]};
                uarray_t<2> qubits_sorted;
                if (gate.affect_args_[0] < gate.affect_args_[1]) {
                    qubits_sorted[0] = gate.affect_args_[0];
                    qubits_sorted[1] = gate.affect_args_[1];
                } else {
                    qubits_sorted[1] = gate.affect_args_[0];
                    qubits_sorted[0] = gate.affect_args_[1];
                }

                if (qubits_sorted[1] == q_state_bit_num - 1) { // ...q
                    if (qubits_sorted[0] == q_state_bit_num - 2) { // ...qq
                        if (qubits_sorted[0] == qubits[0]) { // ...01
                            constexpr uint64_t batch_size = 4;

                            {
#pragma omp parallel for num_threads(DEFAULT_NUM_THREADS) schedule(dynamic, omp_chunk_size(q_state_bit_num))
                                for (uint64_t i = 0; i < (1 << q_state_bit_num); i += batch_size) {
                                    __m256d v_re = _mm256_loadu_pd(real + i);
                                    __m256d v_im = _mm256_loadu_pd(imag + i);
                                    __m256d tmp_re[4], tmp_im[4];

                                    for (int row = 0; row < 4; row++) {
                                        __m256d op_re = _mm256_loadu_pd(gate.mat_real_ + (row << 2));
                                        __m256d op_im = _mm256_loadu_pd(gate.mat_imag_ + (row << 2));
                                        COMPLEX_YMM_MUL(op_re, op_im, v_re, v_im, tmp_re[row], tmp_im[row]);
                                    }

                                    tmp_re[0] = _mm256_hadd_pd(tmp_re[0], tmp_re[2]);
                                    tmp_im[0] = _mm256_hadd_pd(tmp_im[0], tmp_im[2]);
                                    tmp_re[1] = _mm256_hadd_pd(tmp_re[1], tmp_re[3]);
                                    tmp_im[1] = _mm256_hadd_pd(tmp_im[1], tmp_im[3]);
                                    tmp_re[0] = _mm256_permute4x64_pd(tmp_re[0], 0b1101'1000);
                                    tmp_im[0] = _mm256_permute4x64_pd(tmp_im[0], 0b1101'1000);
                                    tmp_re[1] = _mm256_permute4x64_pd(tmp_re[1], 0b1101'1000);
                                    tmp_im[1] = _mm256_permute4x64_pd(tmp_im[1], 0b1101'1000);
                                    tmp_re[0] = _mm256_hadd_pd(tmp_re[0], tmp_re[1]);
                                    tmp_im[0] = _mm256_hadd_pd(tmp_im[0], tmp_im[1]);

                                    _mm256_storeu_pd(real + i, tmp_re[0]);
                                    _mm256_storeu_pd(imag + i, tmp_im[0]);
                                }
                            }
                        } else { // ...10
                            Precision mat_real_[16], mat_imag_[16];
                            for (int row = 0; row < 4; row++) {
                                __m256d ymm0 = _mm256_loadu_pd(gate.mat_real_ + (row << 2));
                                __m256d ymm1 = _mm256_loadu_pd(gate.mat_imag_ + (row << 2));
                                ymm0 = _mm256_permute4x64_pd(ymm0, 0b1101'1000);
                                ymm1 = _mm256_permute4x64_pd(ymm1, 0b1101'1000);
                                _mm256_storeu_pd(mat_real_ + (row << 2), ymm0);
                                _mm256_storeu_pd(mat_imag_ + (row << 2), ymm1);
                            }

                            constexpr uint64_t batch_size = 4;

                            {
#pragma omp parallel for num_threads(DEFAULT_NUM_THREADS) schedule(dynamic, omp_chunk_size(q_state_bit_num))
                                for (uint64_t i = 0; i < (1 << q_state_bit_num); i += batch_size) {
                                    __m256d v_re = _mm256_loadu_pd(real + i);
                                    __m256d v_im = _mm256_loadu_pd(imag + i);
                                    __m256d tmp_re[4], tmp_im[4];

                                    for (int row = 0; row < 4; row++) {
                                        __m256d op_re = _mm256_loadu_pd(mat_real_ + (row << 2));
                                        __m256d op_im = _mm256_loadu_pd(mat_imag_ + (row << 2));
                                        COMPLEX_YMM_MUL(op_re, op_im, v_re, v_im, tmp_re[row], tmp_im[row]);
                                    }

                                    tmp_re[0] = _mm256_hadd_pd(tmp_re[0], tmp_re[1]);
                                    tmp_im[0] = _mm256_hadd_pd(tmp_im[0], tmp_im[1]);
                                    tmp_re[1] = _mm256_hadd_pd(tmp_re[2], tmp_re[3]);
                                    tmp_im[1] = _mm256_hadd_pd(tmp_im[2], tmp_im[3]);
                                    tmp_re[0] = _mm256_permute4x64_pd(tmp_re[0], 0b1101'1000);
                                    tmp_im[0] = _mm256_permute4x64_pd(tmp_im[0], 0b1101'1000);
                                    tmp_re[1] = _mm256_permute4x64_pd(tmp_re[1], 0b1101'1000);
                                    tmp_im[1] = _mm256_permute4x64_pd(tmp_im[1], 0b1101'1000);
                                    tmp_re[0] = _mm256_hadd_pd(tmp_re[0], tmp_re[1]);
                                    tmp_im[0] = _mm256_hadd_pd(tmp_im[0], tmp_im[1]);

                                    _mm256_storeu_pd(real + i, tmp_re[0]);
                                    _mm256_storeu_pd(imag + i, tmp_im[0]);
                                }
                            }
                        }
                    } else { // ...q.q
                        if (qubits_sorted[0] == qubits[0]) { // ...0.1
                            constexpr uint64_t batch_size = 2;
                            uint64_t task_size = 1 << (q_state_bit_num - 2);

                            {
#pragma omp parallel for num_threads(DEFAULT_NUM_THREADS) schedule(dynamic, omp_chunk_size(q_state_bit_num))
                                for (uint64_t task_id = 0; task_id < task_size; task_id += batch_size) {
                                    auto idx = index(task_id, q_state_bit_num, qubits, qubits_sorted);
                                    __m256d v01_re = _mm256_loadu_pd(real + idx[0]);
                                    __m256d v23_re = _mm256_loadu_pd(real + idx[2]);
                                    __m256d v01_im = _mm256_loadu_pd(imag + idx[0]);
                                    __m256d v23_im = _mm256_loadu_pd(imag + idx[2]);

                                    for (int row = 0; row < 4; row += 2) {
                                        __m256d a01_re[2], a01_im[2], a23_re[2], a23_im[2];
                                        __m256d tmp_re, tmp_im;

                                        for (int i = 0; i < 2; i++) {
                                            a01_re[i] = _mm256_loadu2_m128d(TWICE(gate.mat_real_ + ((row + i) << 2)));
                                            a01_im[i] = _mm256_loadu2_m128d(TWICE(gate.mat_imag_ + ((row + i) << 2)));
                                            COMPLEX_YMM_MUL(a01_re[i], a01_im[i], v01_re, v01_im, tmp_re, tmp_im);
                                            a01_re[i] = tmp_re;
                                            a01_im[i] = tmp_im;

                                            a23_re[i] = _mm256_loadu2_m128d(
                                                    TWICE(gate.mat_real_ + (((row + i) << 2) + 2)));
                                            a23_im[i] = _mm256_loadu2_m128d(
                                                    TWICE(gate.mat_imag_ + (((row + i) << 2) + 2)));
                                            COMPLEX_YMM_MUL(a23_re[i], a23_im[i], v23_re, v23_im, tmp_re, tmp_im);
                                            a23_re[i] = tmp_re;
                                            a23_im[i] = tmp_im;
                                        }

                                        a01_re[0] = _mm256_hadd_pd(a01_re[0], a23_re[0]);
                                        a01_re[1] = _mm256_hadd_pd(a01_re[1], a23_re[1]);
                                        tmp_re = _mm256_hadd_pd(a01_re[0], a01_re[1]);
                                        _mm256_storeu_pd(real + idx[row], tmp_re);

                                        a01_im[0] = _mm256_hadd_pd(a01_im[0], a23_im[0]);
                                        a01_im[1] = _mm256_hadd_pd(a01_im[1], a23_im[1]);
                                        tmp_im = _mm256_hadd_pd(a01_im[0], a01_im[1]);
                                        _mm256_storeu_pd(imag + idx[row], tmp_im);
                                    }
                                }
                            }
                        } else { // ...1.0
                            Precision mat_real_[16], mat_imag_[16];
                            for (int row = 0; row < 4; row++) {
                                __m256d ymm0 = _mm256_loadu_pd(gate.mat_real_ + (row << 2));
                                __m256d ymm1 = _mm256_loadu_pd(gate.mat_imag_ + (row << 2));
                                ymm0 = _mm256_permute4x64_pd(ymm0, 0b1101'1000);
                                ymm1 = _mm256_permute4x64_pd(ymm1, 0b1101'1000);
                                _mm256_storeu_pd(mat_real_ + (row << 2), ymm0);
                                _mm256_storeu_pd(mat_imag_ + (row << 2), ymm1);
                            }

                            constexpr uint64_t batch_size = 2;
                            uint64_t task_size = 1 << (q_state_bit_num - 2);

                            {
#pragma omp parallel for num_threads(DEFAULT_NUM_THREADS) schedule(dynamic, omp_chunk_size(q_state_bit_num))
                                for (uint64_t task_id = 0; task_id < task_size; task_id += batch_size) {
                                    auto idx = index(task_id, q_state_bit_num, qubits, qubits_sorted);
                                    __m256d v02_re = _mm256_loadu_pd(real + idx[0]);
                                    __m256d v13_re = _mm256_loadu_pd(real + idx[1]);
                                    __m256d v02_im = _mm256_loadu_pd(imag + idx[0]);
                                    __m256d v13_im = _mm256_loadu_pd(imag + idx[1]);

                                    for (int row = 0; row < 2; row++) {
                                        __m256d a02_re[2], a02_im[2], a13_re[2], a13_im[2];
                                        __m256d tmp_re, tmp_im;

                                        for (int i = 0; i < 2; i++) {
                                            a02_re[i] = _mm256_loadu2_m128d(TWICE(mat_real_ + ((row + (i << 1)) << 2)));
                                            a02_im[i] = _mm256_loadu2_m128d(TWICE(mat_imag_ + ((row + (i << 1)) << 2)));
                                            COMPLEX_YMM_MUL(a02_re[i], a02_im[i], v02_re, v02_im, tmp_re, tmp_im);
                                            a02_re[i] = tmp_re;
                                            a02_im[i] = tmp_im;

                                            a13_re[i] = _mm256_loadu2_m128d(
                                                    TWICE(mat_real_ + (((row + (i << 1)) << 2) + 2)));
                                            a13_im[i] = _mm256_loadu2_m128d(
                                                    TWICE(mat_imag_ + (((row + (i << 1)) << 2) + 2)));
                                            COMPLEX_YMM_MUL(a13_re[i], a13_im[i], v13_re, v13_im, tmp_re, tmp_im);
                                            a13_re[i] = tmp_re;
                                            a13_im[i] = tmp_im;
                                        }

                                        a02_re[0] = _mm256_hadd_pd(a02_re[0], a13_re[0]);
                                        a02_re[1] = _mm256_hadd_pd(a02_re[1], a13_re[1]);
                                        tmp_re = _mm256_hadd_pd(a02_re[0], a02_re[1]);
                                        _mm256_storeu_pd(real + idx[row], tmp_re);

                                        a02_im[0] = _mm256_hadd_pd(a02_im[0], a13_im[0]);
                                        a02_im[1] = _mm256_hadd_pd(a02_im[1], a13_im[1]);
                                        tmp_im = _mm256_hadd_pd(a02_im[0], a02_im[1]);
                                        _mm256_storeu_pd(imag + idx[row], tmp_im);
                                    }
                                }
                            }
                        }
                    }
                } else if (qubits_sorted[1] == q_state_bit_num - 2) { // ...q.
                    if (qubits_sorted[0] == qubits[0]) { // ...0.
                        Precision mat01_real_[16], mat01_imag_[16], mat23_real_[16], mat23_imag_[16];
                        for (int i = 0; i < 4; i++)
                            for (int j = 0; j < 4; j++) {
                                mat01_real_[(i << 2) + j] = gate.mat_real_[(i << 2) + (j >> 1)];
                                mat01_imag_[(i << 2) + j] = gate.mat_imag_[(i << 2) + (j >> 1)];
                                mat23_real_[(i << 2) + j] = gate.mat_real_[(i << 2) + (j >> 1) + 2];
                                mat23_imag_[(i << 2) + j] = gate.mat_imag_[(i << 2) + (j >> 1) + 2];
                            }

                        constexpr uint64_t batch_size = 2;
                        uint64_t task_size = 1 << (q_state_bit_num - 2);

                        {
#pragma omp parallel for num_threads(DEFAULT_NUM_THREADS) schedule(dynamic, omp_chunk_size(q_state_bit_num))
                            for (uint64_t task_id = 0; task_id < task_size; task_id += batch_size) {
                                auto idx = index(task_id, q_state_bit_num, qubits, qubits_sorted);
                                __m256d v01_re = _mm256_loadu_pd(real + idx[0]);
                                __m256d v23_re = _mm256_loadu_pd(real + idx[2]);
                                __m256d v01_im = _mm256_loadu_pd(imag + idx[0]);
                                __m256d v23_im = _mm256_loadu_pd(imag + idx[2]);

                                for (int row = 0; row < 4; row += 2) {
                                    __m256d a01_re[2], a01_im[2], a23_re[2], a23_im[2];
                                    __m256d tmp_re, tmp_im;
                                    for (int i = 0; i < 2; i++) {
                                        a01_re[i] = _mm256_loadu_pd(mat01_real_ + ((row + i) << 2));
                                        a01_im[i] = _mm256_loadu_pd(mat01_imag_ + ((row + i) << 2));
                                        COMPLEX_YMM_MUL(a01_re[i], a01_im[i], v01_re, v01_im, tmp_re, tmp_im);
                                        a01_re[i] = tmp_re;
                                        a01_im[i] = tmp_im;

                                        a23_re[i] = _mm256_loadu_pd(mat23_real_ + ((row + i) << 2));
                                        a23_im[i] = _mm256_loadu_pd(mat23_imag_ + ((row + i) << 2));
                                        COMPLEX_YMM_MUL(a23_re[i], a23_im[i], v23_re, v23_im, tmp_re, tmp_im);
                                        a23_re[i] = tmp_re;
                                        a23_im[i] = tmp_im;

                                        a01_re[i] = _mm256_add_pd(a01_re[i], a23_re[i]);
                                        a01_im[i] = _mm256_add_pd(a01_im[i], a23_im[i]);
                                        a01_re[i] = _mm256_permute4x64_pd(a01_re[i], 0b1101'1000);
                                        a01_im[i] = _mm256_permute4x64_pd(a01_im[i], 0b1101'1000);
                                    }

                                    a01_re[0] = _mm256_hadd_pd(a01_re[0], a01_re[1]);
                                    a01_im[0] = _mm256_hadd_pd(a01_im[0], a01_im[1]);
                                    a01_re[0] = _mm256_permute4x64_pd(a01_re[0], 0b1101'1000);
                                    a01_im[0] = _mm256_permute4x64_pd(a01_im[0], 0b1101'1000);
                                    _mm256_storeu_pd(real + idx[row], a01_re[0]);
                                    _mm256_storeu_pd(imag + idx[row], a01_im[0]);
                                }
                            }
                        }
                    } else { // ...1.
                        Precision mat02_real_[16], mat02_imag_[16], mat13_real_[16], mat13_imag_[16];
                        for (int i = 0; i < 4; i++)
                            for (int j = 0; j < 4; j++) {
                                mat02_real_[(i << 2) + j] = gate.mat_real_[(i << 2) + ((j >> 1) << 1)];
                                mat02_imag_[(i << 2) + j] = gate.mat_imag_[(i << 2) + ((j >> 1) << 1)];
                                mat13_real_[(i << 2) + j] = gate.mat_real_[(i << 2) + ((j >> 1) << 1) + 1];
                                mat13_imag_[(i << 2) + j] = gate.mat_imag_[(i << 2) + ((j >> 1) << 1) + 1];
                            }

                        constexpr uint64_t batch_size = 2;
                        uint64_t task_size = 1 << (q_state_bit_num - 2);

                        {
#pragma omp parallel for num_threads(DEFAULT_NUM_THREADS) schedule(dynamic, omp_chunk_size(q_state_bit_num))
                            for (uint64_t task_id = 0; task_id < task_size; task_id += batch_size) {
                                auto idx = index(task_id, q_state_bit_num, qubits, qubits_sorted);

                                __m256d v02_re = _mm256_loadu_pd(real + idx[0]);
                                __m256d v13_re = _mm256_loadu_pd(real + idx[1]);
                                __m256d v02_im = _mm256_loadu_pd(imag + idx[0]);
                                __m256d v13_im = _mm256_loadu_pd(imag + idx[1]);

                                for (int row = 0; row < 2; row++) {
                                    __m256d a02_re[2], a02_im[2], a13_re[2], a13_im[2];
                                    __m256d tmp_re, tmp_im;
                                    for (int i = 0; i < 2; i++) {
                                        a02_re[i] = _mm256_loadu_pd(mat02_real_ + ((row + (i << 1)) << 2));
                                        a02_im[i] = _mm256_loadu_pd(mat02_imag_ + ((row + (i << 1)) << 2));
                                        COMPLEX_YMM_MUL(a02_re[i], a02_im[i], v02_re, v02_im, tmp_re, tmp_im);
                                        a02_re[i] = tmp_re;
                                        a02_im[i] = tmp_im;

                                        a13_re[i] = _mm256_loadu_pd(mat13_real_ + ((row + (i << 1)) << 2));
                                        a13_im[i] = _mm256_loadu_pd(mat13_imag_ + ((row + (i << 1)) << 2));
                                        COMPLEX_YMM_MUL(a13_re[i], a13_im[i], v13_re, v13_im, tmp_re, tmp_im);
                                        a13_re[i] = tmp_re;
                                        a13_im[i] = tmp_im;

                                        a02_re[i] = _mm256_add_pd(a02_re[i], a13_re[i]);
                                        a02_im[i] = _mm256_add_pd(a02_im[i], a13_im[i]);
                                        a02_re[i] = _mm256_permute4x64_pd(a02_re[i], 0b1101'1000);
                                        a02_im[i] = _mm256_permute4x64_pd(a02_im[i], 0b1101'1000);
                                    }

                                    a02_re[0] = _mm256_hadd_pd(a02_re[0], a02_re[1]);
                                    a02_im[0] = _mm256_hadd_pd(a02_im[0], a02_im[1]);
                                    a02_re[0] = _mm256_permute4x64_pd(a02_re[0], 0b1101'1000);
                                    a02_im[0] = _mm256_permute4x64_pd(a02_im[0], 0b1101'1000);
                                    _mm256_storeu_pd(real + idx[row], a02_re[0]);
                                    _mm256_storeu_pd(imag + idx[row], a02_im[0]);
                                }
                            }
                        }
                    }
                } else { // xxx..
                    constexpr uint64_t batch_size = 4;
                    uint64_t task_size = 1 << (q_state_bit_num - 2);

                    {
#pragma omp parallel for num_threads(DEFAULT_NUM_THREADS) schedule(dynamic, omp_chunk_size(q_state_bit_num))
                        for (uint64_t task_id = 0; task_id < task_size; task_id += batch_size) {
                            auto idx = index(task_id, q_state_bit_num, qubits, qubits_sorted);
                            __m256d v_re[4], v_im[4];
                            for (int i = 0; i < 4; i++) {
                                v_re[i] = _mm256_loadu_pd(real + idx[i]);
                                v_im[i] = _mm256_loadu_pd(imag + idx[i]);
                            }

                            for (int i = 0; i < 4; i++) {
                                __m256d acc_re = _mm256_set1_pd(0);
                                __m256d acc_im = _mm256_set1_pd(0);
                                for (int j = 0; j < 4; j++) {
                                    __m256d op_re = _mm256_set1_pd(gate.mat_real_[(i << 2) + j]);
                                    __m256d op_im = _mm256_set1_pd(gate.mat_imag_[(i << 2) + j]);
                                    __m256d res_re, res_im;
                                    COMPLEX_YMM_MUL(op_re, op_im, v_re[j], v_im[j], res_re, res_im);
                                    acc_re = _mm256_add_pd(acc_re, res_re);
                                    acc_im = _mm256_add_pd(acc_im, res_im);
                                }

                                _mm256_storeu_pd(real + idx[i], acc_re);
                                _mm256_storeu_pd(imag + idx[i], acc_im);
                            }
                        }
                    }
                }
            } else {
                throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": "
                                         + "Not Implemented " + __func__);
            }
        }
#undef TWICE
    }

//
//    template<typename Precision>
//    template<uint64_t N, template<uint64_t, typename> class Gate>
//    void MaTricksSimulator<Precision>::apply_unitary_n_gate(
//            uint64_t circuit_qubit_num,
//            const Gate<N, Precision> &gate,
//            Precision *real,
//            Precision *imag
//    ) {
//#define TWICE(x) (x), (x)
//        if constexpr(std::is_same_v<Precision, float>) {
//            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": "
//                                     + "Not Implemented " + __func__);
//        } else if constexpr(std::is_same_v<Precision, double>) {
//            if constexpr(N == 1) {
//                uint64_t task_num = 1ULL << (circuit_qubit_num - 1);
//                if (gate.targ_ == circuit_qubit_num - 1) {
//                    // op0 := {a00, a01, a00, a01}
//                    // op1 := {a10, a11, a10, a11}
//                    __m256d op_re[2], op_im[2];
//                    for (int i = 0; i < 2; i++) {
//                        op_re[i] = _mm256_setr_pd(gate.mat_real_[i << 1], gate.mat_real_[(i << 1) | 1],
//                                                  gate.mat_real_[i << 1], gate.mat_real_[(i << 1) | 1]);
//                        op_im[i] = _mm256_setr_pd(gate.mat_imag_[i << 1], gate.mat_imag_[(i << 1) | 1],
//                                                  gate.mat_imag_[i << 1], gate.mat_imag_[(i << 1) | 1]);
//                    }
//
//                    constexpr uint64_t batch_size = 4;
//                    for (int i = 0; i < (1 << circuit_qubit_num); i += batch_size) {
//                        __m256d re = _mm256_loadu_pd(&real[i]);
//                        __m256d im = _mm256_loadu_pd(&imag[i]);
//                        __m256d res_re[2], res_im[2];
//                        COMPLEX_YMM_MUL(re, im, op_re[0], op_im[0], res_re[0], res_im[0]);
//                        COMPLEX_YMM_MUL(re, im, op_re[1], op_im[1], res_re[1], res_im[1]);
//
//                        re = _mm256_hadd_pd(res_re[0], res_re[1]);
//                        im = _mm256_hadd_pd(res_im[0], res_im[1]);
//                        _mm256_storeu_pd(&real[i], re);
//                        _mm256_storeu_pd(&imag[i], im);
//                    }
//                } else if (gate.targ_ == circuit_qubit_num - 2) {
//                    __m256d op_re[2], op_im[2];
//                    for (int i = 0; i < 2; i++) {
//                        op_re[i] = _mm256_setr_pd(gate.mat_real_[i << 1], gate.mat_real_[i << 1],
//                                                  gate.mat_real_[(i << 1) | 1], gate.mat_real_[(i << 1) | 1]);
//                        op_im[i] = _mm256_setr_pd(gate.mat_imag_[i << 1], gate.mat_imag_[i << 1],
//                                                  gate.mat_imag_[(i << 1) | 1], gate.mat_imag_[(i << 1) | 1]);
//                    }
//
//                    constexpr uint64_t batch_size = 4;
//                    for (int i = 0; i < (1 << circuit_qubit_num); i += batch_size) {
//                        __m256d re = _mm256_loadu_pd(&real[i]);
//                        __m256d im = _mm256_loadu_pd(&imag[i]);
//                        __m256d res_re[2], res_im[2];
//
//                        COMPLEX_YMM_MUL(re, im, op_re[0], op_im[0], res_re[0], res_im[0]);
//                        COMPLEX_YMM_MUL(re, im, op_re[1], op_im[1], res_re[1], res_im[1]);
//
//                        res_re[0] = _mm256_permute4x64_pd(res_re[0], 0b1101'1000);
//                        res_re[1] = _mm256_permute4x64_pd(res_re[1], 0b1101'1000);
//                        res_im[0] = _mm256_permute4x64_pd(res_im[0], 0b1101'1000);
//                        res_im[1] = _mm256_permute4x64_pd(res_im[1], 0b1101'1000);
//
//                        re = _mm256_hadd_pd(res_re[0], res_re[1]);
//                        im = _mm256_hadd_pd(res_im[0], res_im[1]);
//                        re = _mm256_permute4x64_pd(re, 0b1101'1000);
//                        im = _mm256_permute4x64_pd(im, 0b1101'1000);
//                        _mm256_storeu_pd(&real[i], re);
//                        _mm256_storeu_pd(&imag[i], im);
//                    }
//                } else {
//                    constexpr uint64_t batch_size = 4;
//                    __m256d op_re[4], op_im[4];
//                    for (int i = 0; i < 4; i++) {
//                        op_re[i] = _mm256_set1_pd(gate.mat_real_[i]);
//                        op_im[i] = _mm256_set1_pd(gate.mat_imag_[i]);
//                    }
//
//                    for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
//                        auto ind = index(task_id, circuit_qubit_num, gate.targ_);
//                        __m256d i0_re = _mm256_loadu_pd(&real[ind[0]]);
//                        __m256d i0_im = _mm256_loadu_pd(&imag[ind[0]]);
//                        __m256d i1_re = _mm256_loadu_pd(&real[ind[1]]);
//                        __m256d i1_im = _mm256_loadu_pd(&imag[ind[1]]);
//
//                        __m256d r0_re, r0_im, r1_re, r1_im;
//                        COMPLEX_YMM_MUL(i0_re, i0_im, op_re[0], op_im[0], r0_re, r0_im);
//                        COMPLEX_YMM_MUL(i1_re, i1_im, op_re[1], op_im[1], r1_re, r1_im);
//                        r0_re = _mm256_add_pd(r0_re, r1_re);
//                        r0_im = _mm256_add_pd(r0_im, r1_im);
//
//                        _mm256_storeu_pd(&real[ind[0]], r0_re);
//                        _mm256_storeu_pd(&imag[ind[0]], r0_im);
//
//                        COMPLEX_YMM_MUL(i0_re, i0_im, op_re[2], op_im[2], r0_re, r0_im);
//                        COMPLEX_YMM_MUL(i1_re, i1_im, op_re[3], op_im[3], r1_re, r1_im);
//                        r0_re = _mm256_add_pd(r0_re, r1_re);
//                        r0_im = _mm256_add_pd(r0_im, r1_im);
//
//                        _mm256_storeu_pd(&real[ind[1]], r0_re);
//                        _mm256_storeu_pd(&imag[ind[1]], r0_im);
//                    }
//                }
//            } else if constexpr (N == 2) {
//                uarray_t<2> qubits = {gate.affect_args_[0], gate.affect_args_[1]};
//                uarray_t<2> qubits_sorted;
//                if (gate.affect_args_[0] < gate.affect_args_[1]) {
//                    qubits_sorted[0] = gate.affect_args_[0];
//                    qubits_sorted[1] = gate.affect_args_[1];
//                } else {
//                    qubits_sorted[1] = gate.affect_args_[0];
//                    qubits_sorted[0] = gate.affect_args_[1];
//                }
//
//                if (qubits_sorted[1] == circuit_qubit_num - 1) { // ...q
//                    if (qubits_sorted[0] == circuit_qubit_num - 2) { // ...qq
//                        if (qubits_sorted[0] == qubits[0]) { // ...01
//                            constexpr uint64_t batch_size = 4;
//                            for (uint64_t i = 0; i < (1 << circuit_qubit_num); i += batch_size) {
//                                __m256d v_re = _mm256_loadu_pd(real + i);
//                                __m256d v_im = _mm256_loadu_pd(imag + i);
//                                __m256d tmp_re[4], tmp_im[4];
//
//                                for (int row = 0; row < 4; row++) {
//                                    __m256d op_re = _mm256_loadu_pd(gate.mat_real_ + (row << 2));
//                                    __m256d op_im = _mm256_loadu_pd(gate.mat_imag_ + (row << 2));
//                                    COMPLEX_YMM_MUL(op_re, op_im, v_re, v_im, tmp_re[row], tmp_im[row]);
//                                }
//
//                                tmp_re[0] = _mm256_hadd_pd(tmp_re[0], tmp_re[2]);
//                                tmp_im[0] = _mm256_hadd_pd(tmp_im[0], tmp_im[2]);
//                                tmp_re[1] = _mm256_hadd_pd(tmp_re[1], tmp_re[3]);
//                                tmp_im[1] = _mm256_hadd_pd(tmp_im[1], tmp_im[3]);
//                                tmp_re[0] = _mm256_permute4x64_pd(tmp_re[0], 0b1101'1000);
//                                tmp_im[0] = _mm256_permute4x64_pd(tmp_im[0], 0b1101'1000);
//                                tmp_re[1] = _mm256_permute4x64_pd(tmp_re[1], 0b1101'1000);
//                                tmp_im[1] = _mm256_permute4x64_pd(tmp_im[1], 0b1101'1000);
//                                tmp_re[0] = _mm256_hadd_pd(tmp_re[0], tmp_re[1]);
//                                tmp_im[0] = _mm256_hadd_pd(tmp_im[0], tmp_im[1]);
//
//                                _mm256_storeu_pd(real + i, tmp_re[0]);
//                                _mm256_storeu_pd(imag + i, tmp_im[0]);
//                            }
//                        } else { // ...10
//                            Precision mat_real_[16], mat_imag_[16];
//                            for (int row = 0; row < 4; row++) {
//                                __m256d ymm0 = _mm256_loadu_pd(gate.mat_real_ + (row << 2));
//                                __m256d ymm1 = _mm256_loadu_pd(gate.mat_imag_ + (row << 2));
//                                ymm0 = _mm256_permute4x64_pd(ymm0, 0b1101'1000);
//                                ymm1 = _mm256_permute4x64_pd(ymm1, 0b1101'1000);
//                                _mm256_storeu_pd(mat_real_ + (row << 2), ymm0);
//                                _mm256_storeu_pd(mat_imag_ + (row << 2), ymm1);
//                            }
//
//                            constexpr uint64_t batch_size = 4;
//                            for (uint64_t i = 0; i < (1 << circuit_qubit_num); i += batch_size) {
//                                __m256d v_re = _mm256_loadu_pd(real + i);
//                                __m256d v_im = _mm256_loadu_pd(imag + i);
//                                __m256d tmp_re[4], tmp_im[4];
//
//                                for (int row = 0; row < 4; row++) {
//                                    __m256d op_re = _mm256_loadu_pd(mat_real_ + (row << 2));
//                                    __m256d op_im = _mm256_loadu_pd(mat_imag_ + (row << 2));
//                                    COMPLEX_YMM_MUL(op_re, op_im, v_re, v_im, tmp_re[row], tmp_im[row]);
//                                }
//
//                                tmp_re[0] = _mm256_hadd_pd(tmp_re[0], tmp_re[1]);
//                                tmp_im[0] = _mm256_hadd_pd(tmp_im[0], tmp_im[1]);
//                                tmp_re[1] = _mm256_hadd_pd(tmp_re[2], tmp_re[3]);
//                                tmp_im[1] = _mm256_hadd_pd(tmp_im[2], tmp_im[3]);
//                                tmp_re[0] = _mm256_permute4x64_pd(tmp_re[0], 0b1101'1000);
//                                tmp_im[0] = _mm256_permute4x64_pd(tmp_im[0], 0b1101'1000);
//                                tmp_re[1] = _mm256_permute4x64_pd(tmp_re[1], 0b1101'1000);
//                                tmp_im[1] = _mm256_permute4x64_pd(tmp_im[1], 0b1101'1000);
//                                tmp_re[0] = _mm256_hadd_pd(tmp_re[0], tmp_re[1]);
//                                tmp_im[0] = _mm256_hadd_pd(tmp_im[0], tmp_im[1]);
//
//                                _mm256_storeu_pd(real + i, tmp_re[0]);
//                                _mm256_storeu_pd(imag + i, tmp_im[0]);
//                            }
//                        }
//                    } else { // ...q.q
//                        if (qubits_sorted[0] == qubits[0]) { // ...0.1
//                            constexpr uint64_t batch_size = 2;
//                            uint64_t task_size = 1 << (circuit_qubit_num - 2);
//                            for (uint64_t task_id = 0; task_id < task_size; task_id += batch_size) {
//                                auto idx = index(task_id, circuit_qubit_num, qubits, qubits_sorted);
//                                __m256d v01_re = _mm256_loadu_pd(real + idx[0]);
//                                __m256d v23_re = _mm256_loadu_pd(real + idx[2]);
//                                __m256d v01_im = _mm256_loadu_pd(imag + idx[0]);
//                                __m256d v23_im = _mm256_loadu_pd(imag + idx[2]);
//
//                                for (int row = 0; row < 4; row += 2) {
//                                    __m256d a01_re[2], a01_im[2], a23_re[2], a23_im[2];
//                                    __m256d tmp_re, tmp_im;
//
//                                    for (int i = 0; i < 2; i++) {
//                                        a01_re[i] = _mm256_loadu2_m128d(TWICE(gate.mat_real_ + ((row + i) << 2)));
//                                        a01_im[i] = _mm256_loadu2_m128d(TWICE(gate.mat_imag_ + ((row + i) << 2)));
//                                        COMPLEX_YMM_MUL(a01_re[i], a01_im[i], v01_re, v01_im, tmp_re, tmp_im);
//                                        a01_re[i] = tmp_re;
//                                        a01_im[i] = tmp_im;
//
//                                        a23_re[i] = _mm256_loadu2_m128d(TWICE(gate.mat_real_ + (((row + i) << 2) + 2)));
//                                        a23_im[i] = _mm256_loadu2_m128d(TWICE(gate.mat_imag_ + (((row + i) << 2) + 2)));
//                                        COMPLEX_YMM_MUL(a23_re[i], a23_im[i], v23_re, v23_im, tmp_re, tmp_im);
//                                        a23_re[i] = tmp_re;
//                                        a23_im[i] = tmp_im;
//                                    }
//
//                                    a01_re[0] = _mm256_hadd_pd(a01_re[0], a23_re[0]);
//                                    a01_re[1] = _mm256_hadd_pd(a01_re[1], a23_re[1]);
//                                    tmp_re = _mm256_hadd_pd(a01_re[0], a01_re[1]);
//                                    _mm256_storeu_pd(real + idx[row], tmp_re);
//
//                                    a01_im[0] = _mm256_hadd_pd(a01_im[0], a23_im[0]);
//                                    a01_im[1] = _mm256_hadd_pd(a01_im[1], a23_im[1]);
//                                    tmp_im = _mm256_hadd_pd(a01_im[0], a01_im[1]);
//                                    _mm256_storeu_pd(imag + idx[row], tmp_im);
//                                }
//                            }
//                        } else { // ...1.0
//                            Precision mat_real_[16], mat_imag_[16];
//                            for (int row = 0; row < 4; row++) {
//                                __m256d ymm0 = _mm256_loadu_pd(gate.mat_real_ + (row << 2));
//                                __m256d ymm1 = _mm256_loadu_pd(gate.mat_imag_ + (row << 2));
//                                ymm0 = _mm256_permute4x64_pd(ymm0, 0b1101'1000);
//                                ymm1 = _mm256_permute4x64_pd(ymm1, 0b1101'1000);
//                                _mm256_storeu_pd(mat_real_ + (row << 2), ymm0);
//                                _mm256_storeu_pd(mat_imag_ + (row << 2), ymm1);
//                            }
//
//                            constexpr uint64_t batch_size = 2;
//                            uint64_t task_size = 1 << (circuit_qubit_num - 2);
//                            for (uint64_t task_id = 0; task_id < task_size; task_id += batch_size) {
//                                auto idx = index(task_id, circuit_qubit_num, qubits, qubits_sorted);
//                                __m256d v02_re = _mm256_loadu_pd(real + idx[0]);
//                                __m256d v13_re = _mm256_loadu_pd(real + idx[1]);
//                                __m256d v02_im = _mm256_loadu_pd(imag + idx[0]);
//                                __m256d v13_im = _mm256_loadu_pd(imag + idx[1]);
//
//                                for (int row = 0; row < 2; row++) {
//                                    __m256d a02_re[2], a02_im[2], a13_re[2], a13_im[2];
//                                    __m256d tmp_re, tmp_im;
//
//                                    for (int i = 0; i < 2; i++) {
//                                        a02_re[i] = _mm256_loadu2_m128d(TWICE(mat_real_ + ((row + (i << 1)) << 2)));
//                                        a02_im[i] = _mm256_loadu2_m128d(TWICE(mat_imag_ + ((row + (i << 1)) << 2)));
//                                        COMPLEX_YMM_MUL(a02_re[i], a02_im[i], v02_re, v02_im, tmp_re, tmp_im);
//                                        a02_re[i] = tmp_re;
//                                        a02_im[i] = tmp_im;
//
//                                        a13_re[i] = _mm256_loadu2_m128d(
//                                                TWICE(mat_real_ + (((row + (i << 1)) << 2) + 2)));
//                                        a13_im[i] = _mm256_loadu2_m128d(
//                                                TWICE(mat_imag_ + (((row + (i << 1)) << 2) + 2)));
//                                        COMPLEX_YMM_MUL(a13_re[i], a13_im[i], v13_re, v13_im, tmp_re, tmp_im);
//                                        a13_re[i] = tmp_re;
//                                        a13_im[i] = tmp_im;
//                                    }
//
//                                    a02_re[0] = _mm256_hadd_pd(a02_re[0], a13_re[0]);
//                                    a02_re[1] = _mm256_hadd_pd(a02_re[1], a13_re[1]);
//                                    tmp_re = _mm256_hadd_pd(a02_re[0], a02_re[1]);
//                                    _mm256_storeu_pd(real + idx[row], tmp_re);
//
//                                    a02_im[0] = _mm256_hadd_pd(a02_im[0], a13_im[0]);
//                                    a02_im[1] = _mm256_hadd_pd(a02_im[1], a13_im[1]);
//                                    tmp_im = _mm256_hadd_pd(a02_im[0], a02_im[1]);
//                                    _mm256_storeu_pd(imag + idx[row], tmp_im);
//                                }
//                            }
//                        }
//                    }
//                } else if (qubits_sorted[1] == circuit_qubit_num - 2) { // ...q.
//                    if (qubits_sorted[0] == qubits[0]) { // ...0.
//                        Precision mat01_real_[16], mat01_imag_[16], mat23_real_[16], mat23_imag_[16];
//                        for (int i = 0; i < 4; i++)
//                            for (int j = 0; j < 4; j++) {
//                                mat01_real_[(i << 2) + j] = gate.mat_real_[(i << 2) + (j >> 1)];
//                                mat01_imag_[(i << 2) + j] = gate.mat_imag_[(i << 2) + (j >> 1)];
//                                mat23_real_[(i << 2) + j] = gate.mat_real_[(i << 2) + (j >> 1) + 2];
//                                mat23_imag_[(i << 2) + j] = gate.mat_imag_[(i << 2) + (j >> 1) + 2];
//                            }
//
//                        constexpr uint64_t batch_size = 2;
//                        uint64_t task_size = 1 << (circuit_qubit_num - 2);
//                        for (uint64_t task_id = 0; task_id < task_size; task_id += batch_size) {
//                            auto idx = index(task_id, circuit_qubit_num, qubits, qubits_sorted);
//                            __m256d v01_re = _mm256_loadu_pd(real + idx[0]);
//                            __m256d v23_re = _mm256_loadu_pd(real + idx[2]);
//                            __m256d v01_im = _mm256_loadu_pd(imag + idx[0]);
//                            __m256d v23_im = _mm256_loadu_pd(imag + idx[2]);
//
//                            for (int row = 0; row < 4; row += 2) {
//                                __m256d a01_re[2], a01_im[2], a23_re[2], a23_im[2];
//                                __m256d tmp_re, tmp_im;
//                                for (int i = 0; i < 2; i++) {
//                                    a01_re[i] = _mm256_loadu_pd(mat01_real_ + ((row + i) << 2));
//                                    a01_im[i] = _mm256_loadu_pd(mat01_imag_ + ((row + i) << 2));
//                                    COMPLEX_YMM_MUL(a01_re[i], a01_im[i], v01_re, v01_im, tmp_re, tmp_im);
//                                    a01_re[i] = tmp_re;
//                                    a01_im[i] = tmp_im;
//
//                                    a23_re[i] = _mm256_loadu_pd(mat23_real_ + ((row + i) << 2));
//                                    a23_im[i] = _mm256_loadu_pd(mat23_imag_ + ((row + i) << 2));
//                                    COMPLEX_YMM_MUL(a23_re[i], a23_im[i], v23_re, v23_im, tmp_re, tmp_im);
//                                    a23_re[i] = tmp_re;
//                                    a23_im[i] = tmp_im;
//
//                                    a01_re[i] = _mm256_add_pd(a01_re[i], a23_re[i]);
//                                    a01_im[i] = _mm256_add_pd(a01_im[i], a23_im[i]);
//                                    a01_re[i] = _mm256_permute4x64_pd(a01_re[i], 0b1101'1000);
//                                    a01_im[i] = _mm256_permute4x64_pd(a01_im[i], 0b1101'1000);
//                                }
//
//                                a01_re[0] = _mm256_hadd_pd(a01_re[0], a01_re[1]);
//                                a01_im[0] = _mm256_hadd_pd(a01_im[0], a01_im[1]);
//                                a01_re[0] = _mm256_permute4x64_pd(a01_re[0], 0b1101'1000);
//                                a01_im[0] = _mm256_permute4x64_pd(a01_im[0], 0b1101'1000);
//                                _mm256_storeu_pd(real + idx[row], a01_re[0]);
//                                _mm256_storeu_pd(imag + idx[row], a01_im[0]);
//                            }
//                        }
//                    } else { // ...1.
//                        Precision mat02_real_[16], mat02_imag_[16], mat13_real_[16], mat13_imag_[16];
//                        for (int i = 0; i < 4; i++)
//                            for (int j = 0; j < 4; j++) {
//                                mat02_real_[(i << 2) + j] = gate.mat_real_[(i << 2) + ((j >> 1) << 1)];
//                                mat02_imag_[(i << 2) + j] = gate.mat_imag_[(i << 2) + ((j >> 1) << 1)];
//                                mat13_real_[(i << 2) + j] = gate.mat_real_[(i << 2) + ((j >> 1) << 1) + 1];
//                                mat13_imag_[(i << 2) + j] = gate.mat_imag_[(i << 2) + ((j >> 1) << 1) + 1];
//                            }
//
//                        constexpr uint64_t batch_size = 2;
//                        uint64_t task_size = 1 << (circuit_qubit_num - 2);
//                        for (uint64_t task_id = 0; task_id < task_size; task_id += batch_size) {
//                            auto idx = index(task_id, circuit_qubit_num, qubits, qubits_sorted);
//
//                            __m256d v02_re = _mm256_loadu_pd(real + idx[0]);
//                            __m256d v13_re = _mm256_loadu_pd(real + idx[1]);
//                            __m256d v02_im = _mm256_loadu_pd(imag + idx[0]);
//                            __m256d v13_im = _mm256_loadu_pd(imag + idx[1]);
//
//                            for (int row = 0; row < 2; row++) {
//                                __m256d a02_re[2], a02_im[2], a13_re[2], a13_im[2];
//                                __m256d tmp_re, tmp_im;
//                                for (int i = 0; i < 2; i++) {
//                                    a02_re[i] = _mm256_loadu_pd(mat02_real_ + ((row + (i << 1)) << 2));
//                                    a02_im[i] = _mm256_loadu_pd(mat02_imag_ + ((row + (i << 1)) << 2));
//                                    COMPLEX_YMM_MUL(a02_re[i], a02_im[i], v02_re, v02_im, tmp_re, tmp_im);
//                                    a02_re[i] = tmp_re;
//                                    a02_im[i] = tmp_im;
//
//                                    a13_re[i] = _mm256_loadu_pd(mat13_real_ + ((row + (i << 1)) << 2));
//                                    a13_im[i] = _mm256_loadu_pd(mat13_imag_ + ((row + (i << 1)) << 2));
//                                    COMPLEX_YMM_MUL(a13_re[i], a13_im[i], v13_re, v13_im, tmp_re, tmp_im);
//                                    a13_re[i] = tmp_re;
//                                    a13_im[i] = tmp_im;
//
//                                    a02_re[i] = _mm256_add_pd(a02_re[i], a13_re[i]);
//                                    a02_im[i] = _mm256_add_pd(a02_im[i], a13_im[i]);
//                                    a02_re[i] = _mm256_permute4x64_pd(a02_re[i], 0b1101'1000);
//                                    a02_im[i] = _mm256_permute4x64_pd(a02_im[i], 0b1101'1000);
//                                }
//
//                                a02_re[0] = _mm256_hadd_pd(a02_re[0], a02_re[1]);
//                                a02_im[0] = _mm256_hadd_pd(a02_im[0], a02_im[1]);
//                                a02_re[0] = _mm256_permute4x64_pd(a02_re[0], 0b1101'1000);
//                                a02_im[0] = _mm256_permute4x64_pd(a02_im[0], 0b1101'1000);
//                                _mm256_storeu_pd(real + idx[row], a02_re[0]);
//                                _mm256_storeu_pd(imag + idx[row], a02_im[0]);
//                            }
//                        }
//                    }
//                } else { // xxx..
//                    constexpr uint64_t batch_size = 4;
//                    uint64_t task_size = 1 << (circuit_qubit_num - 2);
//                    for (uint64_t task_id = 0; task_id < task_size; task_id += batch_size) {
//                        auto idx = index(task_id, circuit_qubit_num, qubits, qubits_sorted);
//                        __m256d v_re[4], v_im[4];
//                        for (int i = 0; i < 4; i++) {
//                            v_re[i] = _mm256_loadu_pd(real + idx[i]);
//                            v_im[i] = _mm256_loadu_pd(imag + idx[i]);
//                        }
//
//                        for (int i = 0; i < 4; i++) {
//                            __m256d acc_re = _mm256_set1_pd(0);
//                            __m256d acc_im = _mm256_set1_pd(0);
//                            for (int j = 0; j < 4; j++) {
//                                __m256d op_re = _mm256_set1_pd(gate.mat_real_[(i << 2) + j]);
//                                __m256d op_im = _mm256_set1_pd(gate.mat_imag_[(i << 2) + j]);
//                                __m256d res_re, res_im;
//                                COMPLEX_YMM_MUL(op_re, op_im, v_re[j], v_im[j], res_re, res_im);
//                                acc_re = _mm256_add_pd(acc_re, res_re);
//                                acc_im = _mm256_add_pd(acc_im, res_im);
//                            }
//
//                            _mm256_storeu_pd(real + idx[i], acc_re);
//                            _mm256_storeu_pd(imag + idx[i], acc_im);
//                        }
//                    }
//                }
//            } else {
//                throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": "
//                                         + "Not Implemented " + __func__);
//            }
//        }
//#undef TWICE
//    }

}

#endif //SIM_BACK_AVX_UNITARY_N_GATE_TCC

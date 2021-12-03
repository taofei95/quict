//
// Created by Ci Lei on 2021-11-05.
//

#ifndef SIM_BACK_AVX_MEASURE_GATE_TCC
#define SIM_BACK_AVX_MEASURE_GATE_TCC

#include <cstdint>
#include <algorithm>
#include <vector>
#include <complex>
#include <utility>
#include <string>
#include <omp.h>
#include <immintrin.h>
#include <random>

#include "../gate.h"
#include "../utility.h"
#include "../matricks_simulator.h"

namespace QuICT {
#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"

    template<typename Precision>
    void  MaTricksSimulator<Precision>::apply_measure_gate(
            uint64_t q_state_bit_num,
            const MeasureGate &gate,
            Precision *real,
            Precision *imag
            ) {
        if constexpr(std::is_same_v<Precision, float>) {
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": "
            + "Not Implemented " + __func__);
        } else if constexpr(std::is_same_v<Precision, double>) {
            if(gate.targ_ == q_state_bit_num - 1)
            {
                constexpr uint64_t batch_size = 4;
                double amp_sum = 0;
#pragma omp parallel reduction(+:amp_sum) if(q_state_bit_num > sysconfig_.omp_threshold_) num_threads(sysconfig_.omp_num_thread_)
                {
                    __m256d ymm0 = _mm256_set1_pd(0);
#pragma omp for
                    for(uint64_t i = 0; i < (1 << q_state_bit_num); i += batch_size)
                    {
                        __m256d ymm1 = _mm256_loadu_pd(&real[i]);
                        __m256d ymm2 = _mm256_loadu_pd(&imag[i]);
                        __m256d ymm3;
                        COMPLEX_YMM_NORM(ymm1, ymm2, ymm3);
                        ymm0 = _mm256_add_pd(ymm0, ymm3);
                    }

                    double ymm0_unpack[4];
                    _mm256_storeu_pd(ymm0_unpack, ymm0);
                    amp_sum += ymm0_unpack[0] + ymm0_unpack[2];
                }

                int chosen = random_dist(random_gen) > amp_sum;
                double coef_0 = sqrt(amp_sum), coef_1 = sqrt(1. - amp_sum);
                __m256d ymm0 = chosen ?
                        _mm256_set_pd(1. / coef_1, 0, 1. / coef_1, 0) :
                        _mm256_set_pd(0, 1. / coef_0, 0, 1. / coef_0);

#pragma omp parallel for if(q_state_bit_num > sysconfig_.omp_threshold_) num_threads(sysconfig_.omp_num_thread_)
                for(uint64_t i = 0; i < (1 << q_state_bit_num); i += batch_size)
                {
                    __m256d ymm1 = _mm256_loadu_pd(&real[i]);
                    __m256d ymm2 = _mm256_loadu_pd(&imag[i]);
                    ymm1 = _mm256_mul_pd(ymm1, ymm0);
                    ymm2 = _mm256_mul_pd(ymm2, ymm0);
                    _mm256_storeu_pd(&real[i], ymm1);
                    _mm256_storeu_pd(&imag[i], ymm2);
                }
            }
            else if(gate.targ_ == q_state_bit_num - 2)
            {
                constexpr uint64_t batch_size = 8;
                double amp_sum = 0;
#pragma omp parallel reduction(+:amp_sum) if(q_state_bit_num > sysconfig_.omp_threshold_) num_threads(sysconfig_.omp_num_thread_)
                {
                    __m256d ymm0 = _mm256_set1_pd(0);
#pragma omp for
                    for(uint64_t i = 0; i < (1 << q_state_bit_num); i += batch_size)
                    {
                        __m256d ymm1 = _mm256_loadu2_m128d(&real[i + 4], &real[i]);
                        __m256d ymm2 = _mm256_loadu2_m128d(&imag[i + 4], &imag[i]);
                        __m256d ymm3;
                        COMPLEX_YMM_NORM(ymm1, ymm2, ymm3)
                        ymm0 = _mm256_add_pd(ymm0, ymm3);
                    }

                    double ymm0_unpack[4];
                    _mm256_storeu_pd(ymm0_unpack, ymm0);
                    amp_sum += ymm0_unpack[0] + ymm0_unpack[1] + ymm0_unpack[2] + ymm0_unpack[3];
                }

                int chosen = random_dist(random_gen) > amp_sum;
                double coef_0 = sqrt(amp_sum), coef_1 = sqrt(1. - amp_sum);
                __m256d ymm0 = chosen ?
                        _mm256_set_pd(1. / coef_1, 1. / coef_1, 0, 0) :
                        _mm256_set_pd(0, 0, 1. / coef_0, 1. / coef_0);

#pragma omp parallel for if(q_state_bit_num > sysconfig_.omp_threshold_) num_threads(sysconfig_.omp_num_thread_)
                for(uint64_t i = 0; i < (1 << q_state_bit_num); i += 4)
                {
                    __m256d ymm1 = _mm256_loadu_pd(&real[i]);
                    __m256d ymm2 = _mm256_loadu_pd(&imag[i]);
                    ymm1 = _mm256_mul_pd(ymm1, ymm0);
                    ymm2 = _mm256_mul_pd(ymm2, ymm0);
                    _mm256_storeu_pd(&real[i], ymm1);
                    _mm256_storeu_pd(&imag[i], ymm2);
                }
            }
            else
            {
                // reduction
                constexpr uint64_t batch_size = 4;
                const uint64_t task_size = 1ul << (q_state_bit_num - 1);
                double amp_sum = 0;

#pragma omp parallel reduction(+:amp_sum) if(q_state_bit_num > sysconfig_.omp_threshold_) num_threads(sysconfig_.omp_num_thread_)
                {
                    __m256d ymm0 = _mm256_set1_pd(0);
#pragma omp for
                    for(uint64_t task_id = 0; task_id < task_size; task_id += batch_size)
                    {
                        auto ind = index(task_id, q_state_bit_num, gate.targ_);
                        __m256d ymm1 = _mm256_loadu_pd(&real[ind[0]]);
                        __m256d ymm2 = _mm256_loadu_pd(&imag[ind[0]]);
                        __m256d ymm3;
                        COMPLEX_YMM_NORM(ymm1, ymm2, ymm3);
                        ymm0 = _mm256_add_pd(ymm0, ymm3);
                    }

                    double ymm0_unpack[4];
                    _mm256_storeu_pd(ymm0_unpack, ymm0);
                    amp_sum += ymm0_unpack[0] + ymm0_unpack[1] + ymm0_unpack[2] + ymm0_unpack[3];
                }

                // sample
                int chosen = random_dist(random_gen) > amp_sum;
                double coef_0 = sqrt(amp_sum), coef_1 = sqrt(1. - amp_sum);
                __m256d ymm0 = _mm256_set1_pd(chosen ? coef_1 : coef_0);
                __m256d zero = _mm256_set1_pd(0);

#pragma omp parallel for if(q_state_bit_num > sysconfig_.omp_threshold_) num_threads(sysconfig_.omp_num_thread_)
                for(uint64_t task_id = 0; task_id < task_size; task_id += batch_size)
                {
                    auto ind = index(task_id, q_state_bit_num, gate.targ_);
                    __m256d ymm1 = _mm256_loadu_pd(&real[ind[chosen]]);
                    __m256d ymm2 = _mm256_loadu_pd(&imag[ind[chosen]]);
                    ymm1 = _mm256_div_pd(ymm1, ymm0);
                    ymm2 = _mm256_div_pd(ymm2, ymm0);

                    _mm256_storeu_pd(&real[ind[chosen]], ymm1);
                    _mm256_storeu_pd(&imag[ind[chosen]], ymm2);
                    _mm256_storeu_pd(&real[ind[chosen^1]], zero);
                    _mm256_storeu_pd(&imag[ind[chosen^1]], zero);
                }
            }
        }
    }
}

#endif //SIM_BACK_AVX_MEASURE_GATE_TCC

//
// Created by Ci Lei on 2021-06-23.
//

#ifndef SIMULATION_BACKENDS_SIMULATOR_H
#define SIMULATION_BACKENDS_SIMULATOR_H

#include <cstdint>
#include <algorithm>
#include <vector>
#include <complex>
#include <sstream>
#include <string>
#include <immintrin.h>
#include <omp.h>
#include <thread>
#include "utility.h"
#include "gate.h"

namespace QuICT {

    enum SimulatorMode {
        single,
        batch,
        avx,
        fma
    };


    template<typename precision_t, SimulatorMode sim_mode = SimulatorMode::single>
    class MonoTuneSimulator {
    public:
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        // Constructor & Deconstructor
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        MonoTuneSimulator() {
            auto ss = std::stringstream();
            ss << "MonoTuneSimulator[";
            if constexpr(sim_mode == SimulatorMode::single) {
                ss << "single";
            } else if constexpr(sim_mode == SimulatorMode::batch) {
                ss << "batch";
            } else if constexpr(sim_mode == SimulatorMode::avx) {
                ss << "avx";
            } else if constexpr(sim_mode == SimulatorMode::fma) {
                ss << "fma";
            }
            ss << ",";
            if constexpr(std::is_same<precision_t, float>::value) {
                ss << "float";
            } else if constexpr(std::is_same<precision_t, double>::value) {
                ss << "double";
            }
            ss << "]";
            name_ = ss.str();
        }

        ~MonoTuneSimulator() {
            for (auto &gate_desc:gate_vec_) {
                delete[] gate_desc.data_ptr_;
            }
        }


        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        // Data Access Helper
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        void append_gate(
                std::string qasm_name,
                std::vector<uint64_t> affect_args,
                precision_t parg,
                mat_entry_t<precision_t> *data_ptr
        ) {
            gate_vec_.emplace_back(GateBridgeEntry<precision_t>(
                    qasm_name, affect_args, parg, data_ptr
            ));
        }

        void append_gate(const GateBridgeEntry<precision_t> &gate_desc) {
            gate_vec_.push_back(gate_desc);
        }

        const std::string &name() {
            return name_;
        }

        void set_init_state(std::complex<precision_t> *init_state) {
            state_vector_ = init_state;
        }

        void clear() {
            gate_vec_.clear();
            qubit_num_ = 0;
            state_vector_ = nullptr;
        }

        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        // Run Simulation
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        // Dispatch all gate descriptions to apply_gate
        inline void run(uint64_t qubit_num, std::complex<precision_t> *init_state) {
            qubit_num_ = qubit_num;
            state_vector_ = init_state;


            for (const auto &gate_desc: gate_vec_) {
                if (gate_desc.qasm_name_ == "h") {
                    apply_gate<Gate::HGate<precision_t>>(gate_desc);
                } else if (gate_desc.qasm_name_ == "crz") {
                    apply_gate<Gate::CrzGate<precision_t>>(gate_desc);
                } else {
                    throw std::runtime_error(
                            std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + "Not implemented gate: " +
                            gate_desc.qasm_name_);
                }
            }

            qubit_num_ = 0;
            state_vector_ = nullptr;
        }

        // Transfer gate description into gate
        template<typename gate_t>
        inline void apply_gate(const GateBridgeEntry<precision_t> &gate_desc) {
            using namespace Gate;
            using namespace std;
            if constexpr(is_same<HGate<precision_t>, gate_t>::value) {
                // H gate
                auto gate = HGate<precision_t>(gate_desc.affect_args_[0]);
                apply_gate(gate);
            } else if constexpr(is_same<CrzGate<precision_t>, gate_t>::value) {
                // Crz gate
                auto gate = CrzGate<precision_t>(
                        gate_desc.affect_args_[0],
                        gate_desc.affect_args_[1],
                        gate_desc.parg_);
                apply_gate(gate);
            } else if constexpr(gate_has_mat_repr<gate_t>::value) {
                const auto gate_qubit_num = gate_desc.affect_args_.size();
                switch (gate_qubit_num) {
                    case 1: {
                        auto gate = UnitaryGateN<1, precision_t>(gate_desc.affect_args_.begin(),
                                                                 gate_desc.data_ptr_);
                        apply_gate(gate);
                        break;
                    }
                    case 2: {
                        auto gate = UnitaryGateN<2, precision_t>(gate_desc.affect_args_.begin(),
                                                                 gate_desc.data_ptr_);
                        apply_gate(gate);
                        break;
                    }
                    case 3: {
                        auto gate = UnitaryGateN<3, precision_t>(gate_desc.affect_args_.begin(),
                                                                 gate_desc.data_ptr_);
                        apply_gate(gate);
                        break;
                    }
                    default: {
                        throw std::runtime_error(
                                std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " +
                                "Not implemented for gate larger than 3");
                    }
                }
            } else {
                throw runtime_error("Cannot apply gate for " + gate_desc.qasm_name_);
            }
        }

    protected:
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        // Protected Simulation Methods
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        // Apply gates by sim_mode
        template<class gate_t>
        inline void apply_gate(const gate_t &gate) {
            if constexpr(sim_mode == SimulatorMode::single) {
                uint64_t task_num = 1ULL << (qubit_num_ - Gate::gate_qubit_num<gate_t>::value);


#pragma omp parallel for num_threads(OMP_NPROC)
                for (uint64_t task_id = 0; task_id < task_num; ++task_id) {
                    apply_gate_single_task(task_id, gate);
                }

            } else if constexpr(sim_mode == SimulatorMode::batch) {
                uint64_t task_num = 1ULL << (qubit_num_ - Gate::gate_qubit_num<gate_t>::value);
                constexpr uint64_t batch_size = 4;
                if (task_num < batch_size) {
                    throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " +
                                             "Too few qubits to run in batch mode");
                }


#pragma omp parallel for num_threads(OMP_NPROC)
                for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                    apply_gate_batch_task<batch_size, gate_t>(task_id, gate);
                }

            } else if constexpr(sim_mode == SimulatorMode::avx) {
                uint64_t task_num = 1ULL << (qubit_num_ - Gate::gate_qubit_num<gate_t>::value);
                if (task_num < 32) {
                    throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " +
                                             "Too few qubits to run in avx mode");
                }
                if constexpr(std::is_same<Gate::HGate<precision_t>, gate_t>::value) {
                    constexpr uint64_t batch_size = 2;


#pragma omp parallel for num_threads(OMP_NPROC)
                    for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                        apply_gate_avx_task(task_id, gate);
                    }

                } else if constexpr(std::is_same<Gate::CrzGate<precision_t>, gate_t>::value) {
                    constexpr uint64_t batch_size = 2;


#pragma omp parallel for num_threads(OMP_NPROC)
                    for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                        apply_gate_avx_task(task_id, gate);
                    }

                } else {
                    throw std::runtime_error(
                            std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                            ": " + "Not implemented gate in avx mode");
                }
            } else if constexpr(sim_mode == SimulatorMode::fma) {
                uint64_t task_num = 1ULL << (qubit_num_ - Gate::gate_qubit_num<gate_t>::value);
                if (task_num < 32) {
                    throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " +
                                             "Too few qubits to run in avx mode");
                }
                if constexpr(std::is_same<Gate::HGate<precision_t>, gate_t>::value) {
                    constexpr uint64_t batch_size = 2;


#pragma omp parallel for num_threads(OMP_NPROC)
                    for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                        apply_gate_fma_task(task_id, gate);
                    }

                } else if constexpr(std::is_same<Gate::CrzGate<precision_t>, gate_t>::value) {
                    constexpr uint64_t batch_size = 2;


#pragma omp parallel for num_threads(OMP_NPROC)
                    for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                        apply_gate_fma_task(task_id, gate);
                    }

                } else {
                    throw std::runtime_error(
                            std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                            ": " + "Not implemented gate in fma mode");
                }
            }
        }

        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        // SimulatorMode::single
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        // One task per run.
        inline void apply_gate_single_task(
                const uint64_t task_id,
                const Gate::HGate<precision_t> &gate
        ) {
            auto ind = index(task_id, qubit_num_, gate.targ_);
            auto tmp_arr_1 = marray_t<precision_t, 2>();
            tmp_arr_1[0] = state_vector_[ind[0]];
            tmp_arr_1[1] = state_vector_[ind[1]];
            state_vector_[ind[0]] = gate.sqrt2_inv * tmp_arr_1[0] + gate.sqrt2_inv * tmp_arr_1[1];
            state_vector_[ind[1]] = gate.sqrt2_inv * tmp_arr_1[0] - gate.sqrt2_inv * tmp_arr_1[1];
        }

        inline void apply_gate_single_task(
                const uint64_t task_id,
                const Gate::CrzGate<precision_t> &gate
        ) {
            uarray_t<2> qubits = {gate.carg_, gate.targ_};
            uarray_t<2> qubits_sorted = {gate.carg_, gate.targ_};
            if (qubits_sorted[0] > qubits_sorted[1]) {
                std::swap(qubits_sorted[0], qubits_sorted[1]);
            }
            auto ind = index(task_id, qubit_num_, qubits, qubits_sorted);
            state_vector_[ind[2]] = gate.diagonal_[0] * state_vector_[ind[2]];
            state_vector_[ind[3]] = gate.diagonal_[1] * state_vector_[ind[3]];
        }

        // Default fallback
        template<uint64_t N>
        inline void apply_gate_single_task(
                const uint64_t task_id,
                const Gate::UnitaryGateN<N, precision_t> &gate
        ) {
            if constexpr(N == 1) {
                auto ind = index(task_id, qubit_num_, gate.targ_);
                auto tmp_arr_1 = marray_t<precision_t, 2>();
                tmp_arr_1[0] = state_vector_[ind[0]];
                tmp_arr_1[1] = state_vector_[ind[1]];
                state_vector_[ind[0]] = gate.mat_[0 * 2 + 0] * tmp_arr_1[0] + gate.mat_[0 * 2 + 1] * tmp_arr_1[1];
                state_vector_[ind[1]] = gate.mat_[1 * 2 + 0] * tmp_arr_1[0] + gate.mat_[1 * 2 + 1] * tmp_arr_1[1];
            } else {
                const auto &qubits = gate.affect_args_;
                uarray_t<N> qubits_sorted;
                std::copy(gate.affect_args_.begin(), gate.affect_args_.end(), qubits_sorted);
                auto ind = index(task_id, qubit_num_, qubits, qubits_sorted);
                constexpr uint64_t vec_slice_sz = 1ULL << N;
                marray_t<precision_t, vec_slice_sz> tmp_arr;
#pragma unroll
                for (size_t i = 0; i < vec_slice_sz; ++i) {
                    tmp_arr[i] = state_vector_[ind[i]];
                    state_vector_[ind[i]] = static_cast<mat_entry_t<precision_t>>(0);
                }
#pragma unroll
                for (size_t i = 0; i < vec_slice_sz; ++i) {
#pragma unroll
                    for (size_t j = 0; j < vec_slice_sz; ++j) {
                        state_vector_[ind[i]] += tmp_arr[j] * gate.mat_[j];
                    }
                }
            }
        }

        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        // SimulatorMode::batch
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        template<uint64_t batch_size, class gate_t>
        inline void apply_gate_batch_task(
                const uint64_t task_id,
                const gate_t &gate
        ) {
#pragma unroll
            for (auto i = 0; i < batch_size; ++i) {
                apply_gate_single_task(task_id + i, gate);
            }
        }

        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        // SimulatorMode::avx
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        // !!! [CAUTIONS] !!!
        // EXTREMELY UGLY CODE HERE! BUY YOURSELF SOME MEDICINE IN CASE OF HEART ATTACK!

        inline void apply_gate_avx_task(
                const uint64_t task_id,
                const Gate::HGate<precision_t> &gate
        ) {
            // AVX256 can perform 8 32-bit floating operations or 4 64-bit operations at once
            if constexpr(std::is_same<precision_t, float>::value) {
                throw std::runtime_error("Not implemented for 32-bit avx simulation mode");
            } else if constexpr(std::is_same<precision_t, double>::value) {
                auto ind_a = index(task_id, qubit_num_, gate.targ_);
                auto ind_b = index(task_id + 1, qubit_num_, gate.targ_);
                double v_re[4], v_im[4];

                v_re[0] = state_vector_[ind_a[0]].real();
                v_im[0] = state_vector_[ind_a[0]].imag();

                v_re[1] = state_vector_[ind_a[1]].real();
                v_im[1] = state_vector_[ind_a[1]].imag();

                v_re[2] = state_vector_[ind_b[0]].real();
                v_im[2] = state_vector_[ind_b[0]].imag();

                v_re[3] = state_vector_[ind_b[1]].real();
                v_im[3] = state_vector_[ind_b[1]].imag();

                auto c = gate.sqrt2_inv.real();
                double cc[4] = {c, -c, c, -c};
                double left_re[4], right_re[4], left_im[4], right_im[4];
                double res_re[4], res_im[4];
                left_re[0] = left_re[1] = v_re[0];
                left_re[2] = left_re[3] = v_re[2];
                right_re[0] = right_re[1] = v_re[1];
                right_re[2] = right_re[3] = v_re[3];

                left_im[0] = left_im[1] = v_im[0];
                left_im[2] = left_im[3] = v_im[2];
                right_im[0] = right_im[1] = v_im[1];
                right_im[2] = right_im[3] = v_im[3];

                /*
                 *            ┎         ┒   ┎    ┒      ┎                ┒
                 *            ┃ 1     1 ┃   ┃ v0 ┃      ┃ v0 + ( C * v1) ┃
                 *        C * ┃         ┃ * ┃    ┃   =  ┃                ┃
                 *            ┃ 1    -1 ┃   ┃ v1 ┃      ┃ v0 + (-C * v1) ┃
                 *            ┖         ┚   ┖    ┚      ┖                ┚
                 *            ┎         ┒   ┎    ┒      ┎                ┒
                 *            ┃ 1     1 ┃   ┃ v2 ┃      ┃ v2 + ( C * v3) ┃
                 *        C * ┃         ┃ * ┃    ┃   =  ┃                ┃
                 *            ┃ 1    -1 ┃   ┃ v3 ┃      ┃ v2 + (-C * v3) ┃
                 *            ┖         ┚   ┖    ┚      ┖                ┚
                 *
                 * */

                // real
                _mm256_zeroupper();
                __m256d ymm0 = _mm256_loadu_pd(cc);  // 1/sqrt2 * (1, -1, 1, -1)
                __m256d ymm1 = _mm256_loadu_pd(left_re);
                __m256d ymm2 = _mm256_loadu_pd(right_re);
                __m256d ymm3 = _mm256_mul_pd(ymm0, ymm2);
                ymm3 = _mm256_add_pd(ymm1, ymm3);
                _mm256_storeu_pd(res_re, ymm3);
                // imag
                ymm1 = _mm256_loadu_pd(left_im);
                ymm2 = _mm256_loadu_pd(right_im);
                ymm3 = _mm256_mul_pd(ymm0, ymm2);
                ymm3 = _mm256_add_pd(ymm1, ymm3);
                _mm256_storeu_pd(res_im, ymm3);
                _mm256_zeroupper();
                // combine
                state_vector_[ind_a[0]] = {res_re[0], res_im[0]};
                state_vector_[ind_a[1]] = {res_re[1], res_im[1]};
                state_vector_[ind_b[0]] = {res_re[2], res_im[2]};
                state_vector_[ind_b[1]] = {res_re[3], res_im[3]};
            }
        }

        inline void apply_gate_avx_task(
                const uint64_t task_id,
                const Gate::CrzGate<precision_t> &gate
        ) {
            if constexpr(std::is_same<precision_t, float>::value) {
                throw std::runtime_error("Not implemented for 32-bit avx simulation mode");
            } else if constexpr(std::is_same<precision_t, double>::value) {
                // 1 crz results 2 64-bit op, then 4 64-bit op realize 2 crz

                uarray_t<2> qubits = {gate.carg_, gate.targ_};
                uarray_t<2> qubits_sorted = {gate.carg_, gate.targ_};
                if (qubits_sorted[0] > qubits_sorted[1]) {
                    std::swap(qubits_sorted[0], qubits_sorted[1]);
                }
                auto ind_a = index(task_id, qubit_num_, qubits, qubits_sorted);
                auto ind_b = index(task_id + 1, qubit_num_, qubits, qubits_sorted);
                double d_re[4], d_im[4], v_re[4], v_im[4], res_re[4], res_im[4];

                v_re[0] = state_vector_[ind_a[2]].real();
                v_im[0] = state_vector_[ind_a[2]].imag();

                v_re[1] = state_vector_[ind_a[3]].real();
                v_im[1] = state_vector_[ind_a[3]].imag();

                v_re[2] = state_vector_[ind_b[2]].real();
                v_im[2] = state_vector_[ind_b[2]].imag();

                v_re[3] = state_vector_[ind_b[3]].real();
                v_im[3] = state_vector_[ind_b[3]].imag();

                d_re[0] = d_re[2] = gate.diagonal_[0].real();
                d_im[0] = d_im[2] = gate.diagonal_[0].imag();

                d_re[1] = d_re[3] = gate.diagonal_[1].real();
                d_im[1] = d_im[3] = gate.diagonal_[1].imag();

                /*
                 *            ┎         ┒   ┎    ┒      ┎         ┒
                 *            ┃ d0    0 ┃   ┃ v0 ┃      ┃ d0 * v0 ┃
                 *            ┃         ┃ * ┃    ┃   =  ┃         ┃
                 *            ┃ 0    d1 ┃   ┃ v1 ┃      ┃ d1 * v1 ┃
                 *            ┖         ┚   ┖    ┚      ┖         ┚
                 *            ┎         ┒   ┎    ┒      ┎         ┒
                 *            ┃ d0    0 ┃   ┃ v2 ┃      ┃ d0 * v2 ┃
                 *            ┃         ┃ * ┃    ┃   =  ┃         ┃
                 *            ┃ 0    d1 ┃   ┃ v3 ┃      ┃ d1 * v3 ┃
                 *            ┖         ┚   ┖    ┚      ┖         ┚
                 *
                 *            (a + bj)(c + dj) = (ac - bd) + (ad + bc)j
                 * */

                // d * v == (d_re * v_re - d_im * v_im) + (d_re * v_im + d_im * v_re) * 1j
                _mm256_zeroupper();
                __m256d ymm0 = _mm256_loadu_pd(d_re);
                __m256d ymm1 = _mm256_loadu_pd(d_im);
                __m256d ymm2 = _mm256_loadu_pd(v_re);
                __m256d ymm3 = _mm256_loadu_pd(v_im);
                __m256d ymm4 = _mm256_mul_pd(ymm0, ymm2); // d_re * v_re
                __m256d ymm5 = _mm256_mul_pd(ymm1, ymm3); // d_im * v_im
                __m256d ymm6 = _mm256_mul_pd(ymm0, ymm3); // d_re * v_im
                __m256d ymm7 = _mm256_mul_pd(ymm1, ymm2); // d_im * v_re
                ymm4 = _mm256_sub_pd(ymm4, ymm5);         // d_re * v_re - d_im * v_im
                ymm6 = _mm256_add_pd(ymm6, ymm7);         // d_re * v_im + d_im * v_re
                _mm256_storeu_pd(res_re, ymm4);
                _mm256_storeu_pd(res_im, ymm6);
                _mm256_zeroupper();

                // combine
                state_vector_[ind_a[2]] = {res_re[0], res_im[0]};
                state_vector_[ind_a[3]] = {res_re[1], res_im[1]};
                state_vector_[ind_b[2]] = {res_re[2], res_im[2]};
                state_vector_[ind_b[3]] = {res_re[3], res_im[3]};
            }
        }

        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        // SimulatorMode::fma
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        inline void apply_gate_fma_task(
                const uint64_t task_id,
                const Gate::HGate<precision_t> &gate
        ) {
            if constexpr(std::is_same<precision_t, float>::value) {
                throw std::runtime_error("Not implemented for 32-bit fma simulation mode");
            } else if (std::is_same<precision_t, float>::value) {

                auto ind_a = index(task_id, qubit_num_, gate.targ_);
                auto ind_b = index(task_id + 1, qubit_num_, gate.targ_);
                double v_re[4], v_im[4];

                v_re[0] = state_vector_[ind_a[0]].real();
                v_im[0] = state_vector_[ind_a[0]].imag();

                v_re[1] = state_vector_[ind_a[1]].real();
                v_im[1] = state_vector_[ind_a[1]].imag();

                v_re[2] = state_vector_[ind_b[0]].real();
                v_im[2] = state_vector_[ind_b[0]].imag();

                v_re[3] = state_vector_[ind_b[1]].real();
                v_im[3] = state_vector_[ind_b[1]].imag();

                /*
                 *            ┎         ┒   ┎    ┒      ┎                ┒
                 *            ┃ 1     1 ┃   ┃ v0 ┃      ┃ v0 + ( C * v1) ┃
                 *        C * ┃         ┃ * ┃    ┃   =  ┃                ┃
                 *            ┃ 1    -1 ┃   ┃ v1 ┃      ┃ v0 + (-C * v1) ┃
                 *            ┖         ┚   ┖    ┚      ┖                ┚
                 *            ┎         ┒   ┎    ┒      ┎                ┒
                 *            ┃ 1     1 ┃   ┃ v2 ┃      ┃ v2 + ( C * v3) ┃
                 *        C * ┃         ┃ * ┃    ┃   =  ┃                ┃
                 *            ┃ 1    -1 ┃   ┃ v3 ┃      ┃ v2 + (-C * v3) ┃
                 *            ┖         ┚   ┖    ┚      ┖                ┚
                 *
                 * */

                auto c = gate.sqrt2_inv.real();
                double cc[4] = {c, -c, c, -c};
                double left_re[4], right_re[4], left_im[4], right_im[4];
                double res_re[4], res_im[4];
                left_re[0] = left_re[1] = v_re[0];
                left_re[2] = left_re[3] = v_re[2];
                right_re[0] = right_re[1] = v_re[1];
                right_re[2] = right_re[3] = v_re[3];

                left_im[0] = left_im[1] = v_im[0];
                left_im[2] = left_im[3] = v_im[2];
                right_im[0] = right_im[1] = v_im[1];
                right_im[2] = right_im[3] = v_im[3];

                _mm256_zeroupper();

                // real
                __m256d ymm0 = _mm256_loadu_pd(cc);
                __m256d ymm1 = _mm256_loadu_pd(left_re);
                __m256d ymm2 = _mm256_loadu_pd(right_re);
                __m256d ymm3 = _mm256_fmadd_pd(ymm1, ymm2, ymm0);  // left + (cc * right)
                _mm256_storeu_pd(res_re, ymm3);

                // imag
                ymm1 = _mm256_loadu_pd(left_im);
                ymm2 = _mm256_loadu_pd(right_im);
                ymm3 = _mm256_fmadd_pd(ymm1, ymm2, ymm0);  // left + (cc * right)
                _mm256_storeu_pd(res_im, ymm3);

                _mm256_zeroupper();

                // combine
                state_vector_[ind_a[0]] = {res_re[0], res_im[0]};
                state_vector_[ind_a[1]] = {res_re[1], res_im[1]};
                state_vector_[ind_b[0]] = {res_re[2], res_im[2]};
                state_vector_[ind_b[1]] = {res_re[3], res_im[3]};
            }
        }

        inline void apply_gate_fma_task(
                const uint64_t task_id,
                const Gate::CrzGate<precision_t> &gate
        ) {
            if constexpr(std::is_same<precision_t, float>::value) {
                throw std::runtime_error("Not implemented for 32-bit fma simulation mode");
            } else if constexpr(std::is_same<precision_t, double>::value) {
                uarray_t<2> qubits = {gate.carg_, gate.targ_};
                uarray_t<2> qubits_sorted = {gate.carg_, gate.targ_};
                if (qubits_sorted[0] > qubits_sorted[1]) {
                    std::swap(qubits_sorted[0], qubits_sorted[1]);
                }
                auto ind_a = index(task_id, qubit_num_, qubits, qubits_sorted);
                auto ind_b = index(task_id + 1, qubit_num_, qubits, qubits_sorted);
                double d_re[4], d_im[4], v_re[4], v_im[4], res_re[4], res_im[4];

                v_re[0] = state_vector_[ind_a[2]].real();
                v_im[0] = state_vector_[ind_a[2]].imag();

                v_re[1] = state_vector_[ind_a[3]].real();
                v_im[1] = state_vector_[ind_a[3]].imag();

                v_re[2] = state_vector_[ind_b[2]].real();
                v_im[2] = state_vector_[ind_b[2]].imag();

                v_re[3] = state_vector_[ind_b[3]].real();
                v_im[3] = state_vector_[ind_b[3]].imag();

                d_re[0] = d_re[2] = gate.diagonal_[0].real();
                d_im[0] = d_im[2] = gate.diagonal_[0].imag();

                d_re[1] = d_re[3] = gate.diagonal_[1].real();
                d_im[1] = d_im[3] = gate.diagonal_[1].imag();

                /*
                 *            ┎         ┒   ┎    ┒      ┎         ┒
                 *            ┃ d0    0 ┃   ┃ v0 ┃      ┃ d0 * v0 ┃
                 *            ┃         ┃ * ┃    ┃   =  ┃         ┃
                 *            ┃ 0    d1 ┃   ┃ v1 ┃      ┃ d1 * v1 ┃
                 *            ┖         ┚   ┖    ┚      ┖         ┚
                 *            ┎         ┒   ┎    ┒      ┎         ┒
                 *            ┃ d0    0 ┃   ┃ v2 ┃      ┃ d0 * v2 ┃
                 *            ┃         ┃ * ┃    ┃   =  ┃         ┃
                 *            ┃ 0    d1 ┃   ┃ v3 ┃      ┃ d1 * v3 ┃
                 *            ┖         ┚   ┖    ┚      ┖         ┚
                 *
                 *            (a + bj)(c + dj) = (ac - bd) + (ad + bc)j
                 * */

                // d * v == (d_re * v_re - d_im * v_im) + (d_re * v_im + d_im * v_re) * 1j
                _mm256_zeroupper();
//                __m256d ymm0 = _mm256_setzero_pd();
                __m256d ymm1 = _mm256_loadu_pd(d_re);
                __m256d ymm2 = _mm256_loadu_pd(d_im);
                __m256d ymm3 = _mm256_loadu_pd(v_re);
                __m256d ymm4 = _mm256_loadu_pd(v_im);
                __m256d ymm5 = _mm256_mul_pd(ymm1, ymm3);  // d_re * v_re
                ymm5 = _mm256_fnmadd_pd(ymm2, ymm4, ymm5); // d_re * v_re - d_im * v_im
                __m256d ymm6 = _mm256_mul_pd(ymm1, ymm4);  // d_re * v_im
                ymm6 = _mm256_fmadd_pd(ymm2, ymm3, ymm6);  // d_re * v_im + d_im * v_re
                _mm256_storeu_pd(res_re, ymm5);
                _mm256_storeu_pd(res_im, ymm6);
                _mm256_zeroupper();

                // combine
                state_vector_[ind_a[2]] = {res_re[0], res_im[0]};
                state_vector_[ind_a[3]] = {res_re[1], res_im[1]};
                state_vector_[ind_b[2]] = {res_re[2], res_im[2]};
                state_vector_[ind_b[3]] = {res_re[3], res_im[3]};
            }
        }

    protected:
        std::complex<precision_t> *state_vector_ = nullptr;
//        precision_t *state_vec_sep_re_ = nullptr, *state_vec_sep_im_ = nullptr;
        uint64_t qubit_num_ = 0;
        std::vector<GateBridgeEntry<precision_t>> gate_vec_;
        std::string name_;
    };
}

#endif //SIMULATION_BACKENDS_SIMULATOR_H

//
// Created by Ci Lei on 2021-07-02.
//

#ifndef SIM_BACK_HYBRID_SIMULATOR_H
#define SIM_BACK_HYBRID_SIMULATOR_H

#include <cstdint>
#include <algorithm>
#include <vector>
#include <complex>
#include <utility>
#include <string>
#include <omp.h>
#include <immintrin.h>

#include "gate.h"
#include "utility.h"

namespace QuICT {
    template<typename Precision>
    class HybridSimulator {
    protected:
        std::string name_;
    public:
        HybridSimulator() {
            using namespace std;
            if constexpr(!is_same_v<Precision, double> && !is_same_v<Precision, float>) {
                throw runtime_error("HybridSimulator only supports double/float precision.");
            }
            name_ = "HybridSimulator";
            if constexpr(std::is_same_v<Precision, double>) {
                name_ += "[double]";
            } else if (std::is_same_v<Precision, float>) {
                name_ += "[float] ";
            }
        }

        inline const std::string &name() {
            return name_;
        }

        inline void run(
                uint64_t circuit_qubit_num,
                const std::vector<GateDescription<Precision>> &gate_desc_vec,
                const std::complex<Precision> *init_state
        );

        inline std::complex<Precision> *run(
                uint64_t circuit_qubit_num,
                const std::vector<GateDescription<Precision>> &gate_desc_vec
        );

        inline std::pair<Precision *, Precision *> run_without_combine(
                uint64_t circuit_qubit_num,
                const std::vector<GateDescription<Precision>> &gate_desc_vec
        );

    private:
        inline void qubit_num_checker(uint64_t qubit_num);

        inline void run(
                uint64_t circuit_qubit_num,
                const std::vector<GateDescription<Precision>> &gate_desc_vec,
                Precision *real,
                Precision *imag
        );

        inline std::pair<Precision *, Precision *> separate_complex(
                uint64_t circuit_qubit_num,
                const std::complex<Precision> *c_arr
        );

        inline void combine_complex(
                uint64_t circuit_qubit_num,
                const Precision *real,
                const Precision *imag,
                std::complex<Precision> *res
        );

        inline void apply_gate(
                uint64_t circuit_qubit_num,
                const GateDescription<Precision> &gate_desc,
                Precision *real,
                Precision *imag
        );

        template<template<typename ...> class Gate>
        inline void apply_diag_n_gate(
                uint64_t circuit_qubit_num,
                const Gate<Precision> &gate,
                Precision *real,
                Precision *imag
        );

        inline void apply_ctrl_diag_gate(
                uint64_t circuit_qubit_num,
                const ControlledDiagonalGate<Precision> &gate,
                Precision *real,
                Precision *imag
        );

        template<class Gate>
        inline void apply_unitary_n_gate(
                uint64_t circuit_qubit_num,
                const Gate &gate,
                Precision *real,
                Precision *imag
        );

        inline void apply_ctrl_unitary_gate(
                uint64_t circuit_qubit_num,
                const ControlledUnitaryGate<Precision> &gate,
                Precision *real,
                Precision *imag
        );

        inline void apply_h_gate(
                uint64_t circuit_qubit_num,
                const HGate<Precision> &gate,
                Precision *real,
                Precision *imag
        );

        inline void apply_x_gate(
                uint64_t circuit_qubit_num,
                const XGate<Precision> &gate,
                Precision *real,
                Precision *imag
        );
    };

    template<typename Precision>
    inline void HybridSimulator<Precision>::run(
            uint64_t circuit_qubit_num,
            const std::vector<GateDescription<Precision>> &gate_desc_vec,
            const std::complex<Precision> *init_state
    ) {
        qubit_num_checker(circuit_qubit_num);

        auto pr = separate_complex(circuit_qubit_num, init_state);
        auto real = pr.first;
        auto imag = pr.second;
        run(circuit_qubit_num, gate_desc_vec, real, imag);
        combine_complex(circuit_qubit_num, real, imag, init_state);
        delete[] real;
        delete[] imag;
    }

    template<typename Precision>
    inline std::complex<Precision> *HybridSimulator<Precision>::run(
            uint64_t circuit_qubit_num,
            const std::vector<GateDescription<Precision>> &gate_desc_vec
    ) {
        qubit_num_checker(circuit_qubit_num);

        auto len = 1ULL << circuit_qubit_num;
        auto real = new Precision[len];
        auto imag = new Precision[len];
        auto result = new std::complex<Precision>[len];
        std::fill(real, real + len, 0);
        std::fill(imag, imag + len, 0);
        real[0] = 1.0;
        run(circuit_qubit_num, gate_desc_vec, real, imag);
        combine_complex(circuit_qubit_num, real, imag, result);
        delete[] real;
        delete[] imag;
        return result;
    }

    template<typename Precision>
    inline std::pair<Precision *, Precision *>
    HybridSimulator<Precision>::run_without_combine(
            uint64_t circuit_qubit_num,
            const std::vector<GateDescription<Precision>> &gate_desc_vec
    ) {
        qubit_num_checker(circuit_qubit_num);

        auto len = 1ULL << circuit_qubit_num;
        auto real = new Precision[len];
        auto imag = new Precision[len];
        std::fill(real, real + len, 0);
        std::fill(imag, imag + len, 0);
        real[0] = 1.0;
        run(circuit_qubit_num, gate_desc_vec, real, imag);
        return {real, imag};
    }

    template<typename Precision>
    inline void HybridSimulator<Precision>::run(
            uint64_t circuit_qubit_num,
            const std::vector<GateDescription<Precision>> &gate_desc_vec,
            Precision *real,
            Precision *imag
    ) {
        qubit_num_checker(circuit_qubit_num);

        for (const auto &gate_desc: gate_desc_vec) {
            apply_gate(circuit_qubit_num, gate_desc, real, imag);
        }
    }

    template<typename Precision>
    inline void HybridSimulator<Precision>::qubit_num_checker(uint64_t qubit_num) {
        if (qubit_num <= 4) {
            throw std::runtime_error("Only supports circuit with more than 4 qubits!");
        }
    }

    template<typename Precision>
    inline std::pair<Precision *, Precision *>
    HybridSimulator<Precision>::separate_complex(
            uint64_t circuit_qubit_num,
            const std::complex<Precision> *c_arr
    ) {
        auto len = 1ULL << circuit_qubit_num;
        auto ptr = new Precision[len << 1ULL];
        auto real = ptr;
        auto imag = &ptr[len];
        for (uint64_t i = 0; i < len; i += 4) {
            real[i] = c_arr[i].real();
            imag[i] = c_arr[i].imag();

            real[i + 1] = c_arr[i + 1].real();
            imag[i + 1] = c_arr[i + 1].imag();

            real[i + 2] = c_arr[i + 2].real();
            imag[i + 2] = c_arr[i + 2].imag();

            real[i + 3] = c_arr[i + 3].real();
            imag[i + 3] = c_arr[i + 3].imag();
        }
        return {real, imag};
    }

    template<typename Precision>
    inline void HybridSimulator<Precision>::combine_complex(
            uint64_t circuit_qubit_num,
            const Precision *real,
            const Precision *imag,
            std::complex<Precision> *res
    ) {
        auto len = 1ULL << circuit_qubit_num;
        for (uint64_t i = 0; i < len; i += 4) {
            res[i] = {real[i], imag[i]};
            res[i + 1] = {real[i + 1], imag[i + 1]};
            res[i + 2] = {real[i + 2], imag[i + 2]};
            res[i + 3] = {real[i + 3], imag[i + 3]};
        }
//        return res;
    }

    template<typename Precision>
    inline void HybridSimulator<Precision>::apply_gate(
            uint64_t circuit_qubit_num,
            const GateDescription<Precision> &gate_desc,
            Precision *real,
            Precision *imag
    ) {
        if (gate_desc.qasm_name_ == "h") { // Single Bit
            auto gate = HGate<Precision>(gate_desc.affect_args_[0]);
            apply_h_gate(circuit_qubit_num, gate, real, imag);
        } else if (gate_desc.qasm_name_ == "x") {
            auto gate = XGate<Precision>(gate_desc.affect_args_[0]);
            apply_x_gate(circuit_qubit_num, gate, real, imag);
        } else if (gate_desc.qasm_name_ == "crz") { // Two Bit
            auto gate = CrzGate<Precision>(gate_desc.affect_args_[0], gate_desc.affect_args_[1], gate_desc.parg_);
            apply_ctrl_diag_gate(circuit_qubit_num, gate, real, imag);
        } /*else if (gate_desc.qasm_name_ == "u1") {
            auto gate = UnitaryGateN<1, Precision>(gate_desc.affect_args_[0], gate_desc.data_ptr_);
            apply_unitary_n_gate(circuit_qubit_num, gate, real, imag);
        }*/ else { // Not Implemented
            throw std::runtime_error(std::string(__func__) + ": " + "Not implemented gate - " + gate_desc.qasm_name_);
        }
    }

    //**********************************************************************
    // Special simple gates
    //**********************************************************************


    template<typename Precision>
    inline void HybridSimulator<Precision>::apply_h_gate(
            uint64_t circuit_qubit_num,
            const HGate<Precision> &gate,
            Precision *real,
            Precision *imag
    ) {
        if constexpr (std::is_same_v<Precision, float>) { // float
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": "
                                     + "Not Implemented " + __func__);
        } else if constexpr (std::is_same_v<Precision, double>) { // double
            uint64_t task_num = 1ULL << (circuit_qubit_num - 1);
            if (gate.targ_ == circuit_qubit_num - 1) {
                constexpr uint64_t batch_size = 4;
                auto cc = gate.sqrt2_inv.real();
                __m256d ymm0 = _mm256_broadcast_sd(&cc);
#pragma omp parallel for
                for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                    auto ind_0 = index(task_id, circuit_qubit_num, gate.targ_);

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
            } else if (gate.targ_ == circuit_qubit_num - 2) {
                // After some permutations, this is the same with the previous one.
                constexpr uint64_t batch_size = 4;
                auto cc = gate.sqrt2_inv.real();
#pragma omp parallel for
                for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                    auto ind_0 = index(task_id, circuit_qubit_num, gate.targ_);

                    __m256d ymm0 = _mm256_broadcast_sd(&cc);
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
#pragma omp parallel for
                for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                    auto ind_0 = index(task_id, circuit_qubit_num, gate.targ_);

                    // ind_0[i], ind_1[i], ind_2[i], ind_3[i] are continuous in mem
                    __m256d ymm0 = _mm256_broadcast_sd(&cc);           // constant array
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

    template<typename Precision>
    inline void HybridSimulator<Precision>::apply_x_gate(
            uint64_t circuit_qubit_num,
            const XGate<Precision> &gate,
            Precision *real,
            Precision *imag
    ) {
        if constexpr(std::is_same_v<Precision, float>) {
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": "
                                     + "Not Implemented " + __func__);
        } else if constexpr(std::is_same_v<Precision, double>) {
            uint64_t task_num = 1ULL << (circuit_qubit_num - 1);
            if (gate.targ_ == circuit_qubit_num - 1) {
                constexpr uint64_t batch_size = 4;
#pragma omp parallel for
                for (uint64_t ind = 0; ind < (1ULL << circuit_qubit_num); ind += batch_size) {
                    __m256d ymm1 = _mm256_loadu_pd(&real[ind]);
                    __m256d ymm2 = _mm256_loadu_pd(&imag[ind]);

                    ymm1 = _mm256_permute4x64_pd(ymm1, 0b1011'0001);
                    ymm2 = _mm256_permute4x64_pd(ymm2, 0b1011'0001);
                    _mm256_storeu_pd(&real[ind], ymm1);
                    _mm256_storeu_pd(&imag[ind], ymm2);
                }
            } else if (gate.targ_ == circuit_qubit_num - 2) {
                constexpr uint64_t batch_size = 4;
#pragma omp parallel for
                for (uint64_t ind = 0; ind < (1ULL << circuit_qubit_num); ind += batch_size) {
                    __m256d ymm1 = _mm256_loadu_pd(&real[ind]);
                    __m256d ymm2 = _mm256_loadu_pd(&imag[ind]);

                    ymm1 = _mm256_permute4x64_pd(ymm1, 0b0100'1110);
                    ymm2 = _mm256_permute4x64_pd(ymm2, 0b0100'1110);
                    _mm256_storeu_pd(&real[ind], ymm1);
                    _mm256_storeu_pd(&imag[ind], ymm2);
                }
            } else {
                constexpr uint64_t batch_size = 4;
#pragma omp parallel for
                for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                    auto ind_0 = index(task_id, circuit_qubit_num, gate.targ_);
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



    //**********************************************************************
    // Special matrix pattern gates
    //**********************************************************************

    template<typename Precision>
    inline void HybridSimulator<Precision>::apply_ctrl_diag_gate(
            uint64_t circuit_qubit_num,
            const ControlledDiagonalGate<Precision> &gate,
            Precision *real,
            Precision *imag
    ) {
        if constexpr(std::is_same_v<Precision, float>) {
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": "
                                     + "Not Implemented " + __func__);
        } else if constexpr(std::is_same_v<Precision, double>) {
            // Two qubit distribution is much more sophisticated than
            // single qubit's case. So we enumerate index distance instead
            // of targets positions.

            uarray_t<2> qubits = {gate.carg_, gate.targ_};
            uarray_t<2> qubits_sorted = {gate.carg_, gate.targ_};
            if (gate.carg_ > gate.targ_) {
                qubits_sorted[0] = gate.targ_;
                qubits_sorted[1] = gate.carg_;
            }

            uint64_t task_num = 1ULL << (circuit_qubit_num - 2);
            if (qubits_sorted[1] == circuit_qubit_num - 1) {
                if (qubits_sorted[0] == circuit_qubit_num - 2) {
                    __m256d ymm0; // dr
                    __m256d ymm1; // di
                    ymm0 = _mm256_setr_pd(gate.diagonal_real_[0], gate.diagonal_real_[1],
                                          gate.diagonal_real_[0], gate.diagonal_real_[1]);
                    ymm1 = _mm256_setr_pd(gate.diagonal_imag_[0], gate.diagonal_imag_[1],
                                          gate.diagonal_imag_[0], gate.diagonal_imag_[1]);
                    constexpr uint64_t batch_size = 2;
#pragma omp parallel for
                    for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                        __m256d ymm2; // vr
                        __m256d ymm3; // vi
                        __m256d ymm6, ymm7; // tmp reg
                        auto inds = index(task_id, circuit_qubit_num, qubits, qubits_sorted);
                        if (qubits[0] == qubits_sorted[0]) { // ...q0q1
                            // v00 v01 v02 v03 v10 v11 v12 v13
                            ymm2 = _mm256_loadu2_m128d(&real[inds[2] + 4], &real[inds[2]]);
                            ymm3 = _mm256_loadu2_m128d(&imag[inds[2] + 4], &imag[inds[2]]);
                        } else { // ...q1q0
                            // v00 v02 v01 v03 v10 v12 v11 v13
                            STRIDE_2_LOAD_ODD_PD(&real[inds[0]], ymm2, ymm6, ymm7);
                            STRIDE_2_LOAD_ODD_PD(&imag[inds[0]], ymm3, ymm6, ymm7);
                        }
                        __m256d ymm4; // res_r
                        __m256d ymm5; // res_i
                        COMPLEX_YMM_MUL(ymm0, ymm1, ymm2, ymm3, ymm4, ymm5);
                        if (qubits[0] == qubits_sorted[0]) { // ...q0q1
                            // v00 v01 v02 v03 v10 v11 v12 v13
                            _mm256_storeu2_m128d(&real[inds[2] + 4], &real[inds[2]], ymm4);
                            _mm256_storeu2_m128d(&imag[inds[2] + 4], &imag[inds[2]], ymm5);
                        } else { // ...q1q0
                            // v00 v02 v01 v03 v10 v12 v11 v13
                            Precision tmp[4];
                            STRIDE_2_STORE_ODD_PD(&real[inds[0]], ymm4, tmp);
                            STRIDE_2_STORE_ODD_PD(&imag[inds[0]], ymm5, tmp);
                        }
                    }
                } else if (qubits_sorted[0] < circuit_qubit_num - 2) {
                    if (qubits_sorted[0] == qubits[0]) { // ...q0.q1
                        // v00 v01 v10 v11 v02 v03 v12 v13
                        constexpr uint64_t batch_size = 2;
                        Precision c_arr_real[4] =
                                {gate.diagonal_real_[0], gate.diagonal_real_[1],
                                 gate.diagonal_real_[0], gate.diagonal_real_[1]};
                        Precision c_arr_imag[4] =
                                {gate.diagonal_imag_[0], gate.diagonal_imag_[1],
                                 gate.diagonal_imag_[0], gate.diagonal_imag_[1]};
                        __m256d ymm0 = _mm256_loadu_pd(c_arr_real);
                        __m256d ymm1 = _mm256_loadu_pd(c_arr_imag);
#pragma omp parallel for
                        for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                            auto inds = index(task_id, circuit_qubit_num, qubits, qubits_sorted);
                            __m256d ymm2 = _mm256_loadu_pd(&real[inds[2]]);  // vr
                            __m256d ymm3 = _mm256_loadu_pd(&imag[inds[2]]);  // vi
                            __m256d ymm4;  // res_r
                            __m256d ymm5;  // res_i
                            COMPLEX_YMM_MUL(ymm0, ymm1, ymm2, ymm3, ymm4, ymm5);
                            _mm256_storeu_pd(&real[inds[2]], ymm4);
                            _mm256_storeu_pd(&imag[inds[2]], ymm5);
                        }
                    } else { // ...q1.q0
                        // v00 v02 v10 v12 . v01 v03 v11 v13
                        constexpr uint64_t batch_size = 2;
                        __m256d ymm0 = _mm256_setr_pd(gate.diagonal_real_[0], gate.diagonal_real_[1],
                                                      gate.diagonal_real_[0], gate.diagonal_real_[1]);
                        __m256d ymm1 = _mm256_setr_pd(gate.diagonal_imag_[0], gate.diagonal_imag_[1],
                                                      gate.diagonal_imag_[0], gate.diagonal_imag_[1]);
#pragma omp parallel for
                        for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                            auto inds = index(task_id, circuit_qubit_num, qubits, qubits_sorted);
                            __m256d ymm2 = _mm256_loadu_pd(&real[inds[0]]); // v00 v02 v10 v12, real
                            __m256d ymm3 = _mm256_loadu_pd(&real[inds[1]]); // v01 v03 v11 v13, real
                            __m256d ymm4 = _mm256_loadu_pd(&imag[inds[0]]); // v00 v02 v10 v12, imag
                            __m256d ymm5 = _mm256_loadu_pd(&imag[inds[1]]); // v01 v03 v11 v13, imag

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

                            _mm256_storeu_pd(&real[inds[0]], ymm2);
                            _mm256_storeu_pd(&real[inds[1]], ymm3);
                            _mm256_storeu_pd(&imag[inds[0]], ymm4);
                            _mm256_storeu_pd(&imag[inds[1]], ymm5);
                        }
                    }
                }
            } else if (qubits_sorted[1] == circuit_qubit_num - 2) {
                // ...q.q.
                // Test Passed 2021-09-11
                if (qubits[0] == qubits_sorted[0]) { // ...q0.q1.
                    // v00 v10 v01 v11 ... v02 v12 v03 v13
                    constexpr uint64_t batch_size = 2;
                    Precision c_arr_real[4] =
                            {gate.diagonal_real_[0], gate.diagonal_real_[0],
                             gate.diagonal_real_[1], gate.diagonal_real_[1]};
                    Precision c_arr_imag[4] =
                            {gate.diagonal_imag_[0], gate.diagonal_imag_[0],
                             gate.diagonal_imag_[1], gate.diagonal_imag_[1]};
                    __m256d ymm0 = _mm256_loadu_pd(c_arr_real); // dr
                    __m256d ymm1 = _mm256_loadu_pd(c_arr_imag); // di
#pragma omp parallel for
                    for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                        auto inds = index(task_id, circuit_qubit_num, qubits, qubits_sorted);
                        __m256d ymm2 = _mm256_loadu_pd(&real[inds[2]]); // vr
                        __m256d ymm3 = _mm256_loadu_pd(&imag[inds[2]]);  // vi
                        __m256d ymm4, ymm5;
                        COMPLEX_YMM_MUL(ymm0, ymm1, ymm2, ymm3, ymm4, ymm5);
                        _mm256_storeu_pd(&real[inds[2]], ymm4);
                        _mm256_storeu_pd(&imag[inds[2]], ymm5);
                    }
                } else { // ...q1.q0.
                    // v00 v10 v02 v12 ... v01 v11 v03 v13
                    constexpr uint64_t batch_size = 2;
                    __m256d ymm0 = _mm256_setr_pd(gate.diagonal_real_[0], gate.diagonal_real_[0],
                                                  gate.diagonal_real_[1], gate.diagonal_real_[1]); // dr
                    __m256d ymm1 = _mm256_setr_pd(gate.diagonal_imag_[0], gate.diagonal_imag_[0],
                                                  gate.diagonal_imag_[1], gate.diagonal_imag_[1]); //di

#pragma omp parallel for
                    for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                        auto inds = index(task_id, circuit_qubit_num, qubits, qubits_sorted);
                        __m256d ymm2 = _mm256_loadu2_m128d(&real[inds[1] + 2], &real[inds[0] + 2]); // vr
                        __m256d ymm3 = _mm256_loadu2_m128d(&imag[inds[1] + 2], &imag[inds[0] + 2]); // vi
                        __m256d ymm4; // res_r
                        __m256d ymm5; // res_i
                        COMPLEX_YMM_MUL(ymm0, ymm1, ymm2, ymm3, ymm4, ymm5);
                        _mm256_storeu2_m128d(&real[inds[1] + 2], &real[inds[0] + 2], ymm4);
                        _mm256_storeu2_m128d(&imag[inds[1] + 2], &imag[inds[0] + 2], ymm5);
                    }
                }
            } else if (qubits_sorted[1] < circuit_qubit_num - 2) { // ...q...q..
                // Easiest branch :)
                __m256d ymm0 = _mm256_broadcast_sd(&gate.diagonal_real_[0]);
                __m256d ymm1 = _mm256_broadcast_sd(&gate.diagonal_real_[1]);
                __m256d ymm2 = _mm256_broadcast_sd(&gate.diagonal_imag_[0]);
                __m256d ymm3 = _mm256_broadcast_sd(&gate.diagonal_imag_[1]);
                constexpr uint64_t batch_size = 4;
#pragma omp parallel for
                for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                    auto inds = index(task_id, circuit_qubit_num, qubits, qubits_sorted);
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


    template<typename Precision>
    template<template<typename ...> class Gate>
    void HybridSimulator<Precision>::apply_diag_n_gate(
            uint64_t circuit_qubit_num,
            const Gate<Precision> &gate,
            Precision *real,
            Precision *imag
    ) {
        if constexpr(std::is_same_v<Precision, float>) {
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": "
                                     + "Not Implemented " + __func__);
        } else if constexpr(std::is_same_v<Precision, double>) {

        }
    }

    template<typename Precision>
    template<class Gate>
    void HybridSimulator<Precision>::apply_unitary_n_gate(
            uint64_t circuit_qubit_num,
            const Gate &gate,
            Precision *real,
            Precision *imag
    ) {
        if constexpr(std::is_same_v<Precision, float>) {
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": "
                                     + "Not Implemented " + __func__);
        } else if constexpr(std::is_same_v<Precision, double>) {
//            /* ONE BIT CASE
//             *
//             * /res_0\ = /a00, a01\ * /t_0\
//             * \res_1/   \a10, a11/   \t_1/
//             */
//
//            if (QuICT::gate_qubit_num_v<Gate> == 1) {
//                uint64_t task_num = 1ULL << (circuit_qubit_num - 1);
//                if (gate.targ_ == circuit_qubit_num - 1) {
//                    // op0 := {a00, a01, a00, a01}
//                    // op1 := {a10, a11, a10, a11}
//                    __m256d op_re[2], op_im[2];
//                    for (int i = 0; i < 2; i++) {
//                        op_re[i] = _mm256_setr_pd(gate.mat_[i << 1].real(), gate.mat_[(i << 1) | 1].real(),
//                                                  gate.mat_[i << 1].real(), gate.mat_[(i << 1) | 1].real());
//                        op_im[i] = _mm256_setr_pd(gate.mat_[i << 1].imag(), gate.mat_[(i << 1) | 1].imag(),
//                                                  gate.mat_[i << 1].imag(), gate.mat_[(i << 1) | 1].imag());
//                    }
//
//                    constexpr uint64_t batch_size = 4;
//                    for (int i = 0; i < (1 << circuit_qubit_num); i += batch_size) {
//                        /*
//                         * old  := < t0_0, t0_1, t1_0, t1_1 >
//                         * res0 := old * op0
//                         *       = < a00*t0_0, a01*t0_1, a00*t1_0, a01*t1_1 >
//                         * res1 := old * op1
//                         *       = < a10*t0_0, a11*t0_1, a10*t1_0, a11*t1_1 >
//                         * new  := hadd(res0, res1)
//                         *       = < a00*t0_0 + a01*t0_1, a10*t0_0 + a11*t0_1,
//                         *           a00*t1_0 + a01*t1_1, a10*t1_0 + a11*t1_1 >
//                         */
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
//                        op_re[i] = _mm256_setr_pd(gate.mat_[i << 1].real(), gate.mat_[i << 1].real(),
//                                                  gate.mat_[(i << 1) | 1].real(), gate.mat_[(i << 1) | 1].real());
//                        op_im[i] = _mm256_setr_pd(gate.mat_[i << 1].imag(), gate.mat_[i << 1].imag(),
//                                                  gate.mat_[(i << 1) | 1].imag(), gate.mat_[(i << 1) | 1].imag());
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
//                        op_re[i] = _mm256_set1_pd(gate.mat_[i].real());
//                        op_im[i] = _mm256_set1_pd(gate.mat_[i].imag());
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
//            } else {
//                throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": "
//                                         + "Not Implemented " + __func__);
//            }
        }
    }

    template<typename Precision>
    void HybridSimulator<Precision>::apply_ctrl_unitary_gate(
            uint64_t circuit_qubit_num,
            const ControlledUnitaryGate<Precision> &gate,
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

            /*
             * Multiplication Relation:
             * vi2 <-> m0/m2
             * vi3 <-> m1/m3
             * */

            uint64_t task_num = 1ULL << (circuit_qubit_num - 2);
            if (qubits_sorted[1] == circuit_qubit_num - 1) {
                if (qubits_sorted[0] == circuit_qubit_num - 2) {
                    __m256d ymm0, ymm1, ymm2, ymm3;
                    __m256d ymm4, ymm5;
                    ymm4 = _mm256_loadu_pd(gate.mat_real_); // m0 m1 m2 m3, real
                    ymm5 = _mm256_loadu_pd(gate.mat_imag_); // m0 m1 m2 m3, imag
                    ymm4 = _mm256_permute4x64_pd(ymm4, 0b11011000); // m0 m2 m1 m3, real
                    ymm5 = _mm256_permute4x64_pd(ymm5, 0b11011000); // m0 m2 m1 m3, imag

                    ymm0 = _mm256_permute2f128_pd(ymm4, ymm4, 0b000000); // m0 m2 m0 m2, real
                    ymm1 = _mm256_permute2f128_pd(ymm4, ymm4, 0b001001); // m1 m3 m1 m3, real
                    ymm2 = _mm256_permute2f128_pd(ymm5, ymm5, 0b000000); // m0 m2 m0 m2, imag
                    ymm3 = _mm256_permute2f128_pd(ymm5, ymm5, 0b001001); // m1 m3 m1 m3, imag

                    // Now we need 2 vectors of v02 v02 v12 v12, v03 v03 v13 v13
                    constexpr uint64_t batch_size = 2;
                    for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                        auto ind0 = index0(task_id, circuit_qubit_num, qubits, qubits_sorted);
                        __m256d ymm6, ymm7, ymm8, ymm9;

                        if (qubits_sorted[0] == qubits[0]) {
                            // v00 v01 v02 v03 v10 v11 v12 v13
                            ymm4 = _mm256_loadu2_m128d(&real[ind0 + 6], &real[ind0 + 2]); // v02 v03 v12 v13, real
                            ymm5 = _mm256_loadu2_m128d(&imag[ind0 + 6], &imag[ind0 + 2]); // v02 v03 v12 v13, imag
                        } else {
                            // v00 v02 v01 v03 v10 v12 v11 v13
                            STRIDE_2_LOAD_ODD_PD(&real[ind0], ymm4, ymm6, ymm7); // v02 v03 v12 v13, real
                            STRIDE_2_LOAD_ODD_PD(&imag[ind0], ymm5, ymm6, ymm7); // v02 v03 v12 v13, imag
                        }
                        ymm6 = _mm256_shuffle_pd(ymm4, ymm4, 0b0000); // v02 v02 v12 v12, real
                        ymm7 = _mm256_shuffle_pd(ymm4, ymm4, 0b1111); // v03 v03 v13 v13, real
                        ymm8 = _mm256_shuffle_pd(ymm5, ymm5, 0b0000); // v02 v02 v12 v12, imag
                        ymm9 = _mm256_shuffle_pd(ymm5, ymm5, 0b1111); // v02 v02 v12 v12, imag

                        __m256d ymm10; // res_r1
                        __m256d ymm11; // res_r2
                        __m256d ymm12; // res_i1
                        __m256d ymm13; // res_i2
                        COMPLEX_YMM_MUL(ymm0, ymm2, ymm6, ymm8, ymm10, ymm12);
                        COMPLEX_YMM_MUL(ymm1, ymm3, ymm7, ymm9, ymm11, ymm13);
                        ymm10 = _mm256_add_pd(ymm10, ymm11); // res_r
                        ymm12 = _mm256_add_pd(ymm12, ymm13); // res_i
                        if (qubits_sorted[0] == qubits[0]) {
                            // v00 v01 v02 v03 v10 v11 v12 v13
                            _mm256_storeu2_m128d(&real[ind0 + 6], &real[ind0 + 2], ymm10);
                            _mm256_storeu2_m128d(&imag[ind0 + 6], &imag[ind0 + 2], ymm12);
                        } else {
                            // v00 v02 v01 v03 v10 v12 v11 v13
                            Precision res_r[4], res_i[4];
                            STRIDE_2_STORE_ODD_PD(&real[ind0], ymm10, res_r);
                            STRIDE_2_STORE_ODD_PD(&imag[ind0], ymm12, res_i);
                        }
                    }
                } else if (qubits_sorted[0] < circuit_qubit_num - 2) {
                    // Actually copied from above codes
                    // Maybe we can eliminate duplications :(
                    __m256d ymm0, ymm1, ymm2, ymm3;
                    __m256d ymm4, ymm5;
                    ymm4 = _mm256_loadu_pd(gate.mat_real_); // m0 m1 m2 m3, real
                    ymm5 = _mm256_loadu_pd(gate.mat_imag_); // m0 m1 m2 m3, imag
                    ymm4 = _mm256_permute4x64_pd(ymm4, 0b11011000); // m0 m2 m1 m3, real
                    ymm5 = _mm256_permute4x64_pd(ymm5, 0b11011000); // m0 m2 m1 m3, imag

                    ymm0 = _mm256_permute2f128_pd(ymm4, ymm4, 0b000000); // m0 m2 m0 m2, real
                    ymm1 = _mm256_permute2f128_pd(ymm4, ymm4, 0b001001); // m1 m3 m1 m3, real
                    ymm2 = _mm256_permute2f128_pd(ymm5, ymm5, 0b000000); // m0 m2 m0 m2, imag
                    ymm3 = _mm256_permute2f128_pd(ymm5, ymm5, 0b001001); // m1 m3 m1 m3, imag

                    constexpr uint64_t batch_size = 2;
                    for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                        auto inds = index(task_id, circuit_qubit_num, qubits, qubits_sorted);
                        __m256d ymm6, ymm7, ymm8, ymm9;
                        if (qubits_sorted[0] == qubits[0]) { // ...q0.q1
                            // v00 v01 v10 v11 . v02 v03 v12 v13
                            ymm4 = _mm256_loadu_pd(&real[inds[2]]); // v02 v03 v12 v13, real
                            ymm5 = _mm256_loadu_pd(&imag[inds[2]]); // v02 v03 v12 v13, imag
                        } else { // ...q1.q0
                            // v00 v02 v10 v12 . v01 v03 v11 v13
                            ymm6 = _mm256_loadu_pd(&real[inds[0]]);
                            ymm7 = _mm256_loadu_pd(&real[inds[1]]);
                            ymm8 = _mm256_loadu_pd(&imag[inds[0]]);
                            ymm9 = _mm256_loadu_pd(&imag[inds[1]]);
                            ymm4 = _mm256_shuffle_pd(ymm6, ymm7, 0b1111); // v02 v03 v12 v13, real
                            ymm5 = _mm256_shuffle_pd(ymm8, ymm9, 0b1111); // v02 v03 v12 v13, imag
                        }
                        ymm6 = _mm256_shuffle_pd(ymm4, ymm4, 0b0000); // v02 v02 v12 v12, real
                        ymm7 = _mm256_shuffle_pd(ymm4, ymm4, 0b1111); // v03 v03 v13 v13, real
                        ymm8 = _mm256_shuffle_pd(ymm5, ymm5, 0b0000); // v02 v02 v12 v12, imag
                        ymm9 = _mm256_shuffle_pd(ymm5, ymm5, 0b1111); // v02 v02 v12 v12, imag

                        __m256d ymm10; // res_r1
                        __m256d ymm11; // res_r2
                        __m256d ymm12; // res_i1
                        __m256d ymm13; // res_i2
                        COMPLEX_YMM_MUL(ymm0, ymm2, ymm6, ymm8, ymm10, ymm12);
                        COMPLEX_YMM_MUL(ymm1, ymm3, ymm7, ymm9, ymm11, ymm13);
                        ymm10 = _mm256_add_pd(ymm10, ymm11); // res_r
                        ymm12 = _mm256_add_pd(ymm12, ymm13); // res_i
                        if (qubits_sorted[0] == qubits[0]) { // ...q0.q1
                            // v00 v01 v10 v11 . v02 v03 v12 v13
                            _mm256_storeu_pd(&real[inds[2]], ymm10);
                            _mm256_storeu_pd(&imag[inds[2]], ymm12);
                        } else { // ...q1.q0
                            // v00 v02 v10 v12 . v01 v03 v11 v13
                            Precision res_r[4], res_i[4];
                            _mm256_storeu_pd(res_r, ymm10);
                            _mm256_storeu_pd(res_i, ymm12);
                            real[inds[2]] = res_r[0];
                            real[inds[2] + 2] = res_r[1];
                            real[inds[3]] = res_r[2];
                            real[inds[3] + 2] = res_r[3];

                            imag[inds[2]] = res_i[0];
                            imag[inds[2] + 2] = res_i[1];
                            imag[inds[3]] = res_i[2];
                            imag[inds[3] + 2] = res_i[3];
                        }
                    }
                }
            } else if (qubits_sorted[1] == circuit_qubit_num - 2) {
                // ...q.q.

                // We need 2 vectors as [m0 m0 m2 m2], [m1 m1 m3 m3] now.
                __m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11;
                ymm4 = _mm256_loadu_pd(gate.mat_real_); // m0 m1 m2 m3, real
                ymm5 = _mm256_loadu_pd(gate.mat_imag_); // m0 m1 m2 m3, imag

                ymm0 = _mm256_shuffle_pd(ymm4, ymm4, 0b0000); // m0 m0 m2 m2, real
                ymm1 = _mm256_shuffle_pd(ymm4, ymm4, 0b1111); // m1 m1 m3 m3, real
                ymm2 = _mm256_shuffle_pd(ymm5, ymm5, 0b0000); // m0 m0 m2 m2, imag
                ymm3 = _mm256_shuffle_pd(ymm5, ymm5, 0b1111); // m1 m1 m3 m3, imag

                // We need [v02 v12 v02 v12], [v03 v13 v03 v13] now.
                constexpr uint64_t batch_size = 2;
                for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                    auto inds = index(task_id, circuit_qubit_num, qubits, qubits_sorted);
                    if (qubits[0] == qubits_sorted[0]) { // ...q0.q1.
                        // v00 v10 v01 v11 ... v02 v12 v03 v13
                        ymm8 = _mm256_loadu_pd(&real[inds[2]]); // v02 v12 v03 v13, real
                        ymm9 = _mm256_loadu_pd(&imag[inds[2]]); // v02 v12 v03 v13, imag
                        ymm4 = _mm256_permute2f128_pd(ymm8, ymm8, 0b000000); // v02 v12 v02 v12, real
                        ymm5 = _mm256_permute2f128_pd(ymm8, ymm8, 0b001001); // v03 v13 v03 v13, real
                        ymm6 = _mm256_permute2f128_pd(ymm9, ymm9, 0b000000); // v02 v12 v02 v12, imag
                        ymm7 = _mm256_permute2f128_pd(ymm9, ymm9, 0b001001); // v03 v13 v03 v13, imag
                    } else { // ...q1.q0.
                        // v00 v10 v02 v12 ... v01 v11 v03 v13
                        ymm4 = _mm256_loadu2_m128d(&real[inds[2]], &real[inds[2]]); // v02 v12 v02 v12, real
                        ymm5 = _mm256_loadu2_m128d(&real[inds[3]], &real[inds[3]]); // v03 v13 v03 v13, real
                        ymm6 = _mm256_loadu2_m128d(&imag[inds[2]], &imag[inds[2]]); // v02 v12 v02 v12, imag
                        ymm7 = _mm256_loadu2_m128d(&imag[inds[3]], &imag[inds[3]]); // v03 v13 v03 v13, imag
                    }
                    COMPLEX_YMM_MUL(ymm0, ymm2, ymm4, ymm6, ymm8, ymm10);
                    COMPLEX_YMM_MUL(ymm1, ymm3, ymm5, ymm7, ymm9, ymm11);
                    // ymm8: res_r1
                    // ymm10: res_i1
                    // ymm9: res_r2
                    // ymm11: res_i2
                    ymm8 = _mm256_add_pd(ymm8, ymm9); // res_r
                    ymm10 = _mm256_add_pd(ymm10, ymm11); // res_i
                    if (qubits[0] == qubits_sorted[0]) { // ...q0.q1.
                        // v00 v10 v01 v11 ... v02 v12 v03 v13
                        _mm256_storeu_pd(&real[inds[2]], ymm8);
                        _mm256_storeu_pd(&imag[inds[2]], ymm10);
                    } else { // ...q1.q0.
                        // v00 v10 v02 v12 ... v01 v11 v03 v13
                        _mm256_storeu2_m128d(&real[inds[3]], &real[inds[2]], ymm8);
                        _mm256_storeu2_m128d(&imag[inds[3]], &imag[inds[2]], ymm10);
                    }
                }
            } else if (qubits_sorted[1] < circuit_qubit_num - 2) { // ...q...q..
                constexpr uint64_t batch_size = 2;
                __m256d ymm0 = _mm256_loadu2_m128d(&gate.mat_real_[0], &gate.mat_real_[0]); // m0 m1 m0 m1, real
                __m256d ymm1 = _mm256_loadu2_m128d(&gate.mat_real_[2], &gate.mat_real_[2]); // m2 m3 m2 m3, real
                __m256d ymm2 = _mm256_loadu2_m128d(&gate.mat_imag_[0], &gate.mat_imag_[0]); // m0 m1 m0 m1, imag
                __m256d ymm3 = _mm256_loadu2_m128d(&gate.mat_imag_[2], &gate.mat_imag_[2]); // m2 m3 m2 m3, imag

                for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                    auto inds = index(task_id, circuit_qubit_num, qubits, qubits_sorted);
                    __m256d ymm4 = _mm256_loadu2_m128d(&real[inds[3]], &real[inds[2]]); // v02 v12 v03 v13, real
                    __m256d ymm5 = _mm256_loadu2_m128d(&imag[inds[3]], &imag[inds[2]]); // v02 v12 v03 v13, imag
                    ymm4 = _mm256_permute4x64_pd(ymm4, 0b11011000); // v02 v03 v12 v13, real
                    ymm5 = _mm256_permute4x64_pd(ymm5, 0b11011000); // v02 v03 v12 v13, imag
                    __m256d ymm6, ymm7, ymm8, ymm9;
                    COMPLEX_YMM_MUL(ymm0, ymm2, ymm4, ymm5, ymm6, ymm7);
                    COMPLEX_YMM_MUL(ymm1, ymm3, ymm4, ymm5, ymm8, ymm9);
                    ymm4 = _mm256_hadd_pd(ymm6, ymm8); // v02 v03 v12 v13, real
                    ymm5 = _mm256_hadd_pd(ymm7, ymm9); // v02 v03 v12 v13, imag
                    ymm4 = _mm256_permute4x64_pd(ymm4, 0b11011000); // v02 v12 v03 v13, real
                    ymm5 = _mm256_permute4x64_pd(ymm5, 0b11011000); // v02 v12 v03 v13, real
                    _mm256_storeu2_m128d(&real[inds[3]], &real[inds[2]], ymm4);
                    _mm256_storeu2_m128d(&imag[inds[3]], &imag[inds[2]], ymm5);
                }
            }
        }
    }
}

#endif //SIM_BACK_HYBRID_SIMULATOR_H
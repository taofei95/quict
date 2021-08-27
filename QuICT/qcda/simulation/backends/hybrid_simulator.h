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

#include "utility.h"
#include "monotonous_simulator.h"

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

        template<template<typename ...> class Gate>
        inline void apply_ctrl_diag_gate(
                uint64_t circuit_qubit_num,
                const Gate<Precision> &gate,
                Precision *real,
                Precision *imag
        );

        template<template<typename ...> class Gate>
        inline void apply_unitary_n_gate(
                uint64_t circuit_qubit_num,
                const Gate<Precision> &gate,
                Precision *real,
                Precision *imag
        );

        template<template<typename ...> class Gate>
        inline void apply_ctrl_unitary_gate(
                uint64_t circuit_qubit_num,
                const Gate<Precision> &gate,
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
        delete real;
        delete imag;
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
    inline void HybridSimulator<Precision>::run(
            uint64_t circuit_qubit_num,
            const std::vector<GateDescription<Precision>> &gate_desc_vec,
            Precision *real,
            Precision *imag
    ) {
        qubit_num_checker(circuit_qubit_num);

        for (const auto &gate_desc:gate_desc_vec) {
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
        } else if(gate_desc.qasm_name_ == "x") {
            auto gate = XGate<Precision>(gate_desc.affect_args_[0]);
            apply_x_gate(circuit_qubit_num, gate, real, imag);
        } else if (gate_desc.qasm_name_ == "crz") { // Two Bit
            auto gate = CrzGate<Precision>(gate_desc.affect_args_[0], gate_desc.affect_args_[1], gate_desc.parg_);
            apply_ctrl_diag_gate(circuit_qubit_num, gate, real, imag);
        } else { // Not Implemented
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
        if (std::is_same_v<Precision, float>) { // float
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": "
                                     + "Not Implemented " + __func__);
        } else { // double
            uint64_t task_num = 1ULL << (circuit_qubit_num - 1);
            if (gate.targ_ == circuit_qubit_num - 1) {
                constexpr uint64_t batch_size = 4;
                for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                    auto ind_0 = index(task_id, circuit_qubit_num, gate.targ_);
                    auto cc = gate.sqrt2_inv.real();

                    __m256d ymm0 = _mm256_broadcast_sd(&cc);
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
                for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                    auto ind_0 = index(task_id, circuit_qubit_num, gate.targ_);
                    auto cc = gate.sqrt2_inv.real();

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
                for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                    auto ind_0 = index(task_id, circuit_qubit_num, gate.targ_);

                    // ind_0[i], ind_1[i], ind_2[i], ind_3[i] are continuous in mem
                    auto cc = gate.sqrt2_inv.real();
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
            if(gate.targ_ == circuit_qubit_num - 1)
            {
                constexpr uint64_t batch_size = 4;
                for(uint64_t ind = 0; ind < (1ULL << circuit_qubit_num); ind += batch_size)
                {
                    __m256d ymm1 = _mm256_loadu_pd(&real[ind]);
                    __m256d ymm2 = _mm256_loadu_pd(&imag[ind]);

                    ymm1 = _mm256_permute4x64_pd(ymm1, 0b1011'0001);
                    ymm2 = _mm256_permute4x64_pd(ymm2, 0b1011'0001);
                    _mm256_storeu_pd(&real[ind], ymm1);
                    _mm256_storeu_pd(&imag[ind], ymm2);
                }
            }
            else if(gate.targ_ == circuit_qubit_num - 2)
            {
                constexpr uint64_t batch_size = 4;
                for(uint64_t ind = 0; ind < (1ULL << circuit_qubit_num); ind += batch_size)
                {
                    __m256d ymm1 = _mm256_loadu_pd(&real[ind]);
                    __m256d ymm2 = _mm256_loadu_pd(&imag[ind]);

                    ymm1 = _mm256_permute4x64_pd(ymm1, 0b0100'1110);
                    ymm2 = _mm256_permute4x64_pd(ymm2, 0b0100'1110);
                    _mm256_storeu_pd(&real[ind], ymm1);
                    _mm256_storeu_pd(&imag[ind], ymm2);
                }
            }
            else
            {
                constexpr uint64_t batch_size = 4;
                for(uint64_t task_id = 0; task_id < task_num; task_id += batch_size)
                {
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
    template<template<typename ...> class Gate>
    inline void HybridSimulator<Precision>::apply_ctrl_diag_gate(
            uint64_t circuit_qubit_num,
            const Gate<Precision> &gate,
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
            constexpr uint64_t batch_size = 2;
            uint64_t task_num = 1ULL << (circuit_qubit_num - 2);
            uarray_t<2> qubits = {gate.carg_, gate.targ_};
            uarray_t<2> qubits_sorted = {gate.carg_, gate.targ_};
            if (gate.carg_ > gate.targ_) {
                qubits_sorted[0] = gate.targ_;
                qubits_sorted[1] = gate.carg_;
            }
            for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                auto ind_0 = index(task_id, circuit_qubit_num, qubits, qubits_sorted);
                auto ind_1 = index(task_id + 1, circuit_qubit_num, qubits, qubits_sorted);
                Precision res_r[4], res_i[4];
                if (ind_0[2] + 1 == ind_1[2] && ind_0[3] + 1 == ind_1[3]) {
                    __m256d ymm0 = _mm256_loadu2_m128d(gate.diagonal_real_, gate.diagonal_real_);
                    __m256d ymm1 = _mm256_loadu2_m128d(gate.diagonal_imag_, gate.diagonal_imag_);
                    ymm0 = _mm256_permute4x64_pd(ymm0, 0b1101'1000);                    // dr
                    ymm1 = _mm256_permute4x64_pd(ymm1, 0b1101'1000);                    // di
                    __m256d ymm2 = _mm256_loadu2_m128d(&real[ind_0[3]], &real[ind_0[2]]);  // vr
                    __m256d ymm3 = _mm256_loadu2_m128d(&imag[ind_0[3]], &imag[ind_0[2]]);  // vi
                    // v * d == (vr * dr - vi * di) + (vi * dr + vr * di)I
                    __m256d ymm4 = _mm256_mul_pd(ymm2, ymm0);  // vr * dr
                    ymm4 = _mm256_fnmadd_pd(ymm1, ymm3, ymm4); // vr * dr - vi * di
                    __m256d ymm5 = _mm256_mul_pd(ymm3, ymm0);  // vi * dr
                    ymm5 = _mm256_fmadd_pd(ymm2, ymm1, ymm5);  // vi * dr + vr * di

                    // Store
                    _mm256_storeu2_m128d(&real[ind_0[3]], &real[ind_0[2]], ymm4);
                    _mm256_storeu2_m128d(&imag[ind_0[3]], &imag[ind_0[2]], ymm5);
                } else if (ind_0[2] + 1 == ind_0[3] && ind_1[2] + 1 == ind_1[3]) {
                    __m256d ymm0 = _mm256_loadu2_m128d(gate.diagonal_real_, gate.diagonal_real_);  // dr
                    __m256d ymm1 = _mm256_loadu2_m128d(gate.diagonal_imag_, gate.diagonal_imag_);  // di
                    __m256d ymm2 = _mm256_loadu2_m128d(&real[ind_1[2]], &real[ind_0[2]]);          // vr
                    __m256d ymm3 = _mm256_loadu2_m128d(&imag[ind_1[2]], &imag[ind_0[2]]);          // vi
                    // v * d == (vr * dr - vi * di) + (vi * dr + vr * di)I
                    __m256d ymm4 = _mm256_mul_pd(ymm2, ymm0);  // vr * dr
                    ymm4 = _mm256_fnmadd_pd(ymm1, ymm3, ymm4); // vr * dr - vi * di
                    __m256d ymm5 = _mm256_mul_pd(ymm3, ymm0);  // vi * dr
                    ymm5 = _mm256_fmadd_pd(ymm2, ymm1, ymm5);  // vi * dr + vr * di

                    // Store
                    _mm256_storeu2_m128d(&real[ind_1[2]], &real[ind_0[2]], ymm4);
                    _mm256_storeu2_m128d(&imag[ind_1[2]], &imag[ind_0[2]], ymm5);
                } else { // Default fallback
                    __m256d ymm0 = _mm256_loadu2_m128d(gate.diagonal_real_, gate.diagonal_real_);                  // dr
                    __m256d ymm1 = _mm256_loadu2_m128d(gate.diagonal_imag_, gate.diagonal_imag_);                  // di
                    __m256d ymm2 = _mm256_setr_pd(real[ind_0[2]], real[ind_0[3]], real[ind_1[2]], real[ind_1[3]]); // vr
                    __m256d ymm3 = _mm256_setr_pd(imag[ind_0[2]], imag[ind_0[3]], imag[ind_1[2]], imag[ind_1[3]]); // vi
                    // v * d == (vr * dr - vi * di) + (vi * dr + vr * di)I
                    __m256d ymm4 = _mm256_mul_pd(ymm2, ymm0);  // vr * dr
                    ymm4 = _mm256_fnmadd_pd(ymm1, ymm3, ymm4); // vr * dr - vi * di
                    __m256d ymm5 = _mm256_mul_pd(ymm3, ymm0);  // vi * dr
                    ymm5 = _mm256_fmadd_pd(ymm2, ymm1, ymm5);  // vi * dr + vr * di
                    _mm256_storeu_pd(res_r, ymm4);
                    _mm256_storeu_pd(res_i, ymm5);

                    // Store
                    real[ind_0[2]] = res_r[0];
                    real[ind_0[3]] = res_r[1];
                    real[ind_1[2]] = res_r[2];
                    real[ind_1[3]] = res_r[3];

                    imag[ind_0[2]] = res_i[0];
                    imag[ind_0[3]] = res_i[1];
                    imag[ind_1[2]] = res_i[2];
                    imag[ind_1[3]] = res_i[3];
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
    template<template<typename ...> class Gate>
    void HybridSimulator<Precision>::apply_unitary_n_gate(
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
    template<template<typename ...> class Gate>
    void HybridSimulator<Precision>::apply_ctrl_unitary_gate(
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
}

#endif //SIM_BACK_HYBRID_SIMULATOR_H

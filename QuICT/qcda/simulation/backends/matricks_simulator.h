//
// Created by Ci Lei on 2021-07-02.
//

#ifndef SIM_BACK_MATRICKS_SIMULATOR_H
#define SIM_BACK_MATRICKS_SIMULATOR_H

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
    class MaTricksSimulator {
    protected:
        std::string name_;
    public:
        MaTricksSimulator() {
            using namespace std;
            if constexpr(!is_same_v<Precision, double> && !is_same_v<Precision, float>) {
                throw runtime_error("MaTricksSimulator only supports double/float precision.");
            }
            name_ = "MaTricksSimulator";
            if constexpr(std::is_same_v<Precision, double>) {
                name_ += "[double]";
            } else if (std::is_same_v<Precision, float>) {
                name_ += " [float]";
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

        template<uint64_t N>
        inline void apply_diag_n_gate(
                uint64_t circuit_qubit_num,
                const DiagonalGateN<N, Precision> &gate,
                Precision *real,
                Precision *imag
        );

        inline void apply_ctrl_diag_gate(
                uint64_t circuit_qubit_num,
                const ControlledDiagonalGate<Precision> &gate,
                Precision *real,
                Precision *imag
        );

        void apply_unitary_n_gate(
                uint64_t circuit_qubit_num,
                const UnitaryGateN<1, Precision> &gate,
                Precision *real,
                Precision *imag
        );

        template<uint64_t N, template<uint64_t, typename> class Gate>
        void apply_unitary_n_gate(
                uint64_t circuit_qubit_num,
                const Gate<N, Precision> &gate,
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
    inline void MaTricksSimulator<Precision>::run(
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
    inline std::complex<Precision> *MaTricksSimulator<Precision>::run(
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
    MaTricksSimulator<Precision>::run_without_combine(
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
    inline void MaTricksSimulator<Precision>::run(
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
    inline void MaTricksSimulator<Precision>::qubit_num_checker(uint64_t qubit_num) {
        if (qubit_num <= 4) {
            throw std::runtime_error("Only supports circuit with more than 4 qubits!");
        }
    }

    template<typename Precision>
    inline std::pair<Precision *, Precision *>
    MaTricksSimulator<Precision>::separate_complex(
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
    inline void MaTricksSimulator<Precision>::combine_complex(
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
    inline void MaTricksSimulator<Precision>::apply_gate(
            uint64_t circuit_qubit_num,
            const GateDescription<Precision> &gate_desc,
            Precision *real,
            Precision *imag
    ) {
        if (gate_desc.qasm_name_ == "h") {
            auto gate = HGate<Precision>(gate_desc.affect_args_[0]);
            apply_h_gate(circuit_qubit_num, gate, real, imag);
        } else if (gate_desc.qasm_name_ == "x") {
            auto gate = XGate<Precision>(gate_desc.affect_args_[0]);
            apply_x_gate(circuit_qubit_num, gate, real, imag);
        } else if (gate_desc.qasm_name_ == "rz") {
            auto gate = RzGate(gate_desc.affect_args_[0], gate_desc.parg_);
            apply_diag_n_gate(circuit_qubit_num, gate, real, imag);
        } else if (gate_desc.qasm_name_ == "crz") {
            auto gate = CrzGate<Precision>(gate_desc.affect_args_[0], gate_desc.affect_args_[1], gate_desc.parg_);
            apply_ctrl_diag_gate(circuit_qubit_num, gate, real, imag);
        } else if (gate_desc.qasm_name_ == "u1") {
            auto gate = UnitaryGateN<1, Precision>(gate_desc.affect_args_[0], gate_desc.data_ptr_);
            apply_unitary_n_gate(circuit_qubit_num, gate, real, imag);
        } else if (gate_desc.qasm_name_ == "u2") {
            auto gate = UnitaryGateN<2, Precision>(gate_desc.affect_args_, gate_desc.data_ptr_);
            apply_unitary_n_gate(circuit_qubit_num, gate, real, imag);
        } else if (gate_desc.qasm_name_ == "cu3") {
            auto gate = CU3Gate<Precision>(gate_desc.affect_args_[0], gate_desc.affect_args_[1], gate_desc.pargs_);
            apply_ctrl_unitary_gate(circuit_qubit_num, gate, real, imag);
        } else { // Not Implemented
            throw std::runtime_error(std::string(__func__) + ": " + "Not implemented gate - " + gate_desc.qasm_name_);
        }
    }

    //**********************************************************************
    // Special simple gates
    //**********************************************************************


    template<typename Precision>
    inline void MaTricksSimulator<Precision>::apply_h_gate(
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
    inline void MaTricksSimulator<Precision>::apply_x_gate(
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
    inline void MaTricksSimulator<Precision>::apply_ctrl_diag_gate(
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
                        // v00 v01 v10 v11 . v02 v03 v12 v13
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
    template<uint64_t N>
    void MaTricksSimulator<Precision>::apply_diag_n_gate(
            uint64_t circuit_qubit_num,
            const DiagonalGateN<N, Precision> &gate,
            Precision *real,
            Precision *imag
    ) {
        if constexpr(std::is_same_v<Precision, float>) {
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": "
                                     + "Not Implemented " + __func__);
        } else if constexpr(std::is_same_v<Precision, double>) {
            if constexpr(N == 2) {
                throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": "
                                         + "Not Implemented " + __func__);
            } else {
                uint64_t task_num = 1ULL << (circuit_qubit_num - 1);
                if (gate.targ_ == circuit_qubit_num - 1) {
                    __m256d ymm0 = _mm256_loadu2_m128d(gate.diagonal_real_, gate.diagonal_real_); // d_r
                    __m256d ymm1 = _mm256_loadu2_m128d(gate.diagonal_imag_, gate.diagonal_imag_); // d_i
                    constexpr uint64_t batch_size = 2;
                    for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                        auto ind0 = index0(task_id, circuit_qubit_num, gate.targ_);
                        __m256d ymm2 = _mm256_loadu_pd(&real[ind0]); // v_r
                        __m256d ymm3 = _mm256_loadu_pd(&imag[ind0]); // v_i
                        __m256d ymm4, ymm5;
                        COMPLEX_YMM_MUL(ymm0, ymm1, ymm2, ymm3, ymm4, ymm5);
                        _mm256_storeu_pd(&real[ind0], ymm4);
                        _mm256_storeu_pd(&imag[ind0], ymm5);
                    }
                } else if (gate.targ_ == circuit_qubit_num - 2) {
                    __m256d ymm0 = _mm256_loadu2_m128d(gate.diagonal_real_, gate.diagonal_real_);
                    __m256d ymm1 = _mm256_loadu2_m128d(gate.diagonal_imag_, gate.diagonal_imag_);
                    ymm0 = _mm256_permute4x64_pd(ymm0, 0b1101'1000); // d_r
                    ymm1 = _mm256_permute4x64_pd(ymm1, 0b1101'1000); // d_i
                    constexpr uint64_t batch_size = 2;
                    for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                        auto ind0 = index0(task_id, circuit_qubit_num, gate.targ_);
                        __m256d ymm2 = _mm256_loadu_pd(&real[ind0]); // v_r
                        __m256d ymm3 = _mm256_loadu_pd(&imag[ind0]); // v_i
                        __m256d ymm4, ymm5;
                        COMPLEX_YMM_MUL(ymm0, ymm1, ymm2, ymm3, ymm4, ymm5);
                        _mm256_storeu_pd(&real[ind0], ymm4);
                        _mm256_storeu_pd(&imag[ind0], ymm5);
                    }
                } else { // gate.targ_ < circuit_qubit_num - 2
                    __m256d ymm0 = _mm256_broadcast_sd(&gate.diagonal_real_[0]);
                    __m256d ymm1 = _mm256_broadcast_sd(&gate.diagonal_imag_[0]);
                    __m256d ymm2 = _mm256_broadcast_sd(&gate.diagonal_real_[1]);
                    __m256d ymm3 = _mm256_broadcast_sd(&gate.diagonal_imag_[1]);
                    constexpr uint64_t batch_size = 2;
                    for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                        auto inds = index(task_id, circuit_qubit_num, gate.targ_);
                        __m256d ymm4 = _mm256_loadu_pd(&real[inds[0]]); // v00 v10 v20 v30, real
                        __m256d ymm5 = _mm256_loadu_pd(&imag[inds[0]]); // v00 v10 v20 v30, imag
                        __m256d ymm6 = _mm256_loadu_pd(&real[inds[1]]); // v01 v11 v21 v31, real
                        __m256d ymm7 = _mm256_loadu_pd(&imag[inds[1]]); // v01 v11 v21 v31, imag
                        __m256d ymm8, ymm9, ymm10, ymm11;
                        COMPLEX_YMM_MUL(ymm0, ymm1, ymm4, ymm5, ymm8, ymm9);
                        COMPLEX_YMM_MUL(ymm2, ymm3, ymm5, ymm6, ymm10, ymm11);
                        _mm256_storeu_pd(&real[inds[0]], ymm8);
                        _mm256_storeu_pd(&imag[inds[0]], ymm9);
                        _mm256_storeu_pd(&real[inds[1]], ymm10);
                        _mm256_storeu_pd(&imag[inds[1]], ymm11);
                    }
                }
            }
        }
    }

    template<typename Precision>
    void MaTricksSimulator<Precision>::apply_unitary_n_gate(
            uint64_t circuit_qubit_num,
            const UnitaryGateN<1, Precision> &gate,
            Precision *real,
            Precision *imag
    ) {
        uint64_t task_num = 1ULL << (circuit_qubit_num - 1);
        if (gate.targ_ == circuit_qubit_num - 1) {
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
            for (int i = 0; i < (1 << circuit_qubit_num); i += batch_size) {
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
        } else if (gate.targ_ == circuit_qubit_num - 2) {
            __m256d op_re[2], op_im[2];
            for (int i = 0; i < 2; i++) {
                op_re[i] = _mm256_setr_pd(gate.mat_real_[i << 1], gate.mat_real_[i << 1],
                                          gate.mat_real_[(i << 1) | 1], gate.mat_real_[(i << 1) | 1]);
                op_im[i] = _mm256_setr_pd(gate.mat_imag_[i << 1], gate.mat_imag_[i << 1],
                                          gate.mat_imag_[(i << 1) | 1], gate.mat_imag_[(i << 1) | 1]);
            }

            constexpr uint64_t batch_size = 4;
            for (int i = 0; i < (1 << circuit_qubit_num); i += batch_size) {
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
        } else {
            constexpr uint64_t batch_size = 4;
            __m256d op_re[4], op_im[4];
            for (int i = 0; i < 4; i++) {
                op_re[i] = _mm256_set1_pd(gate.mat_real_[i]);
                op_im[i] = _mm256_set1_pd(gate.mat_imag_[i]);
            }

            for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                auto ind = index(task_id, circuit_qubit_num, gate.targ_);
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

    template<typename Precision>
    template<uint64_t N, template<uint64_t, typename> class Gate>
    void MaTricksSimulator<Precision>::apply_unitary_n_gate(
            uint64_t circuit_qubit_num,
            const Gate<N, Precision> &gate,
            Precision *real,
            Precision *imag
    ) {
#define TWICE(x) (x), (x)
        if constexpr(std::is_same_v<Precision, float>) {
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": "
                                     + "Not Implemented " + __func__);
        } else if constexpr(std::is_same_v<Precision, double>) {
            if (N == 2) {
                uarray_t<2> qubits = {gate.affect_args_[0], gate.affect_args_[1]};
                uarray_t<2> qubits_sorted;
                if (gate.affect_args_[0] < gate.affect_args_[1]) {
                    qubits_sorted[0] = gate.affect_args_[0];
                    qubits_sorted[1] = gate.affect_args_[1];
                } else {
                    qubits_sorted[1] = gate.affect_args_[0];
                    qubits_sorted[0] = gate.affect_args_[1];
                }

                if (qubits_sorted[1] == circuit_qubit_num - 1) { // ...q
                    if (qubits_sorted[0] == circuit_qubit_num - 2) { // ...qq
                        if (qubits_sorted[0] == qubits[0]) { // ...01
                            constexpr uint64_t batch_size = 4;
                            for (uint64_t i = 0; i < (1 << circuit_qubit_num); i += batch_size) {
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
                        } else { // ...10
                            Precision mat_real_[16], mat_imag_[16];
                            for(int row = 0; row < 4; row++) {
                                __m256d ymm0 = _mm256_loadu_pd(gate.mat_real_ + (row << 2));
                                __m256d ymm1 = _mm256_loadu_pd(gate.mat_imag_ + (row << 2));
                                ymm0 = _mm256_permute4x64_pd(ymm0, 0b1101'1000);
                                ymm1 = _mm256_permute4x64_pd(ymm1, 0b1101'1000);
                                _mm256_storeu_pd(mat_real_ + (row << 2), ymm0);
                                _mm256_storeu_pd(mat_imag_ + (row << 2), ymm1);
                            }

                            constexpr uint64_t batch_size = 4;
                            for (uint64_t i = 0; i < (1 << circuit_qubit_num); i += batch_size) {
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
                    } else { // ...q.q
                        if (qubits_sorted[0] == qubits[0]) { // ...0.1
                            constexpr uint64_t batch_size = 2;
                            uint64_t task_size = 1 << (circuit_qubit_num - 2);
                            for (uint64_t task_id = 0; task_id < task_size; task_id += batch_size) {
                                auto idx = index(task_id, circuit_qubit_num, qubits, qubits_sorted);
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

                                        a23_re[i] = _mm256_loadu2_m128d(TWICE(gate.mat_real_ + (((row + i) << 2) + 2)));
                                        a23_im[i] = _mm256_loadu2_m128d(TWICE(gate.mat_imag_ + (((row + i) << 2) + 2)));
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
                        } else { // ...1.0
                            Precision mat_real_[16], mat_imag_[16];
                            for(int row = 0; row < 4; row++) {
                                __m256d ymm0 = _mm256_loadu_pd(gate.mat_real_ + (row << 2));
                                __m256d ymm1 = _mm256_loadu_pd(gate.mat_imag_ + (row << 2));
                                ymm0 = _mm256_permute4x64_pd(ymm0, 0b1101'1000);
                                ymm1 = _mm256_permute4x64_pd(ymm1, 0b1101'1000);
                                _mm256_storeu_pd(mat_real_ + (row << 2), ymm0);
                                _mm256_storeu_pd(mat_imag_ + (row << 2), ymm1);
                            }

                            constexpr uint64_t batch_size = 2;
                            uint64_t task_size = 1 << (circuit_qubit_num - 2);
                            for (uint64_t task_id = 0; task_id < task_size; task_id += batch_size) {
                                auto idx = index(task_id, circuit_qubit_num, qubits, qubits_sorted);
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
                } else if (qubits_sorted[1] == circuit_qubit_num - 2) { // ...q.
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
                        uint64_t task_size = 1 << (circuit_qubit_num - 2);
                        for (uint64_t task_id = 0; task_id < task_size; task_id += batch_size) {
                            auto idx = index(task_id, circuit_qubit_num, qubits, qubits_sorted);
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
                        uint64_t task_size = 1 << (circuit_qubit_num - 2);
                        for(uint64_t task_id = 0; task_id < task_size; task_id += batch_size) {
                            auto idx = index(task_id, circuit_qubit_num, qubits, qubits_sorted);

                            __m256d v02_re = _mm256_loadu_pd(real + idx[0]);
                            __m256d v13_re = _mm256_loadu_pd(real + idx[1]);
                            __m256d v02_im = _mm256_loadu_pd(imag + idx[0]);
                            __m256d v13_im = _mm256_loadu_pd(imag + idx[1]);

                            for(int row = 0; row < 2; row ++) {
                                __m256d a02_re[2], a02_im[2], a13_re[2], a13_im[2];
                                __m256d tmp_re, tmp_im;
                                for(int i = 0; i < 2; i ++) {
                                    a02_re[i] = _mm256_loadu_pd(mat02_real_ + ((row+(i<<1))<<2));
                                    a02_im[i] = _mm256_loadu_pd(mat02_imag_ + ((row+(i<<1))<<2));
                                    COMPLEX_YMM_MUL(a02_re[i], a02_im[i], v02_re, v02_im, tmp_re, tmp_im);
                                    a02_re[i] = tmp_re; a02_im[i] = tmp_im;

                                    a13_re[i] = _mm256_loadu_pd(mat13_real_ + ((row+(i<<1))<<2));
                                    a13_im[i] = _mm256_loadu_pd(mat13_imag_ + ((row+(i<<1))<<2));
                                    COMPLEX_YMM_MUL(a13_re[i], a13_im[i], v13_re, v13_im, tmp_re, tmp_im);
                                    a13_re[i] = tmp_re; a13_im[i] = tmp_im;

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
                } else { // xxx..
                    constexpr uint64_t batch_size = 4;
                    uint64_t task_size = 1 << (circuit_qubit_num - 2);
                    for (uint64_t task_id = 0; task_id < task_size; task_id += batch_size) {
                        auto idx = index(task_id, circuit_qubit_num, qubits, qubits_sorted);
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
            } else {
                throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": "
                                         + "Not Implemented " + __func__);
            }
        }
    }

    template<typename Precision>
    void MaTricksSimulator<Precision>::apply_ctrl_unitary_gate(
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
                    __m256d ymm4, ymm5, ymm6, ymm7, ymm8, ymm9;
                    ymm4 = _mm256_loadu_pd(gate.mat_real_); // m0 m1 m2 m3, real
                    ymm5 = _mm256_loadu_pd(gate.mat_imag_); // m0 m1 m2 m3, imag

                    ymm0 = _mm256_permute2f128_pd(ymm4, ymm4, 0b0000'0000); // m0 m1 m0 m1, real
                    ymm1 = _mm256_permute2f128_pd(ymm5, ymm5, 0b0000'0000); // m0 m1 m0 m1, imag
                    ymm2 = _mm256_permute2f128_pd(ymm4, ymm4, 0b0001'0001); // m2 m3 m2 m3, real
                    ymm3 = _mm256_permute2f128_pd(ymm5, ymm5, 0b0001'0001); // m2 m3 m2 m3, imag

                    constexpr uint64_t batch_size = 2;
#pragma omp parallel for firstprivate(ymm0, ymm1, ymm2, ymm3) private(ymm4, ymm5, ymm6, ymm7, ymm8, ymm9)
                    for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                        auto ind0 = index0(task_id, circuit_qubit_num, qubits, qubits_sorted);
                        if (qubits[0] == qubits_sorted[0]) { // ...q0q1
                            // v00 v01 v02 v03 v10 v11 v12 v13
                            ymm4 = _mm256_loadu2_m128d(&real[ind0 + 6], &real[ind0 + 2]);
                            ymm5 = _mm256_loadu2_m128d(&imag[ind0 + 6], &imag[ind0 + 2]);
                        } else { // ...q1q0
                            // v00 v02 v01 v03 v10 v12 v11 v13
                            STRIDE_2_LOAD_ODD_PD(&real[ind0], ymm4, ymm6, ymm7);
                            STRIDE_2_LOAD_ODD_PD(&imag[ind0], ymm5, ymm6, ymm7);
                        }
                        // Now: ymm4 -> v_r, ymm5 -> v_i
                        COMPLEX_YMM_MUL(ymm0, ymm1, ymm4, ymm5, ymm6, ymm7);
                        COMPLEX_YMM_MUL(ymm2, ymm3, ymm4, ymm5, ymm8, ymm9);
                        ymm4 = _mm256_hadd_pd(ymm6, ymm8); // res_r
                        ymm5 = _mm256_hadd_pd(ymm7, ymm9); // res_i

                        if (qubits[0] == qubits_sorted[0]) { // ...q0q1
                            // v00 v01 v02 v03 v10 v11 v12 v13
                            _mm256_storeu2_m128d(&real[ind0 + 6], &real[ind0 + 2], ymm4);
                            _mm256_storeu2_m128d(&imag[ind0 + 6], &imag[ind0 + 2], ymm5);
                        } else { // ...q1q0
                            // v00 v02 v01 v03 v10 v12 v11 v13
                            Precision tmp_r[4], tmp_i[4];
                            STRIDE_2_STORE_ODD_PD(&real[ind0], ymm4, tmp_r);
                            STRIDE_2_STORE_ODD_PD(&imag[ind0], ymm5, tmp_i);
                        }
                    }
                } else if (qubits_sorted[0] < circuit_qubit_num - 2) {
                    // Actually copied from above codes
                    // Maybe we can eliminate duplications :(
                    __m256d ymm0, ymm1, ymm2, ymm3;
                    __m256d ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11;
                    ymm4 = _mm256_loadu_pd(gate.mat_real_); // m0 m1 m2 m3, real
                    ymm5 = _mm256_loadu_pd(gate.mat_imag_); // m0 m1 m2 m3, imag

                    ymm0 = _mm256_permute2f128_pd(ymm4, ymm4, 0b0000'0000); // m0 m1 m0 m1, real
                    ymm1 = _mm256_permute2f128_pd(ymm5, ymm5, 0b0000'0000); // m0 m1 m0 m1, imag
                    ymm2 = _mm256_permute2f128_pd(ymm4, ymm4, 0b0001'0001); // m2 m3 m2 m3, real
                    ymm3 = _mm256_permute2f128_pd(ymm5, ymm5, 0b0001'0001); // m2 m3 m2 m3, imag

                    constexpr uint64_t batch_size = 2;
#pragma omp parallel for firstprivate(ymm0, ymm1, ymm2, ymm3) private(ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11)
                    for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                        auto inds = index(task_id, circuit_qubit_num, qubits, qubits_sorted);
                        if (qubits_sorted[0] == qubits[0]) { // ...q0.q1
                            // v00 v01 v10 v11 . v02 v03 v12 v13
                            ymm4 = _mm256_loadu_pd(&real[inds[2]]); // v_r
                            ymm5 = _mm256_loadu_pd(&imag[inds[2]]); // v_r
                        } else { // ...q1.q0
                            // v00 v02 v10 v12 . v01 v03 v11 v13
                            ymm6 = _mm256_loadu_pd(&real[inds[0]]);
                            ymm7 = _mm256_loadu_pd(&real[inds[1]]);
                            ymm8 = _mm256_loadu_pd(&imag[inds[0]]);
                            ymm9 = _mm256_loadu_pd(&imag[inds[1]]);
                            ymm4 = _mm256_shuffle_pd(ymm6, ymm7, 0b1111); // v02 v03 v12 v13, real
                            ymm5 = _mm256_shuffle_pd(ymm8, ymm9, 0b1111); // v02 v03 v12 v13, imag
                            ymm10 = _mm256_shuffle_pd(ymm6, ymm7, 0b0000); // v00 v01 v10 v11, real
                            ymm11 = _mm256_shuffle_pd(ymm8, ymm9, 0b0000); // v00 v01 v10 v11, imag
                        }
                        // Now: ymm4 -> v_r, ymm5 -> v_i
                        COMPLEX_YMM_MUL(ymm0, ymm1, ymm4, ymm5, ymm6, ymm7);
                        COMPLEX_YMM_MUL(ymm2, ymm3, ymm4, ymm5, ymm8, ymm9);
                        ymm4 = _mm256_hadd_pd(ymm6, ymm8); // res_r
                        ymm5 = _mm256_hadd_pd(ymm7, ymm9); // res_i

                        if (qubits_sorted[0] == qubits[0]) { // ...q0.q1
                            // v00 v01 v10 v11 . v02 v03 v12 v13
                            _mm256_storeu_pd(&real[inds[2]], ymm4);
                            _mm256_storeu_pd(&imag[inds[2]], ymm5);
                        } else { // ...q1.q0
                            // v00 v02 v10 v12 . v01 v03 v11 v13
                            ymm6 = _mm256_shuffle_pd(ymm10, ymm4, 0b0000); // v00 v02 v12 v12, real
                            ymm7 = _mm256_shuffle_pd(ymm11, ymm5, 0b0000); // v00 v02 v12 v12, imag
                            ymm8 = _mm256_shuffle_pd(ymm10, ymm4, 0b1111); // v01 v03 v11 v13, real
                            ymm9 = _mm256_shuffle_pd(ymm11, ymm5, 0b1111); // v01 v03 v11 v13, imag
                            _mm256_storeu_pd(&real[inds[0]], ymm6);
                            _mm256_storeu_pd(&real[inds[1]], ymm8);
                            _mm256_storeu_pd(&imag[inds[0]], ymm7);
                            _mm256_storeu_pd(&imag[inds[1]], ymm9);
                        }
                    }
                }
            } else if (qubits_sorted[1] == circuit_qubit_num - 2) {
                constexpr uint64_t batch_size = 2;
                __m256d ymm0 = _mm256_loadu2_m128d(&gate.mat_real_[0], &gate.mat_real_[0]); // m0 m1 m0 m1, real
                __m256d ymm1 = _mm256_loadu2_m128d(&gate.mat_real_[2], &gate.mat_real_[2]); // m2 m3 m2 m3, real
                __m256d ymm2 = _mm256_loadu2_m128d(&gate.mat_imag_[0], &gate.mat_imag_[0]); // m0 m1 m0 m1, imag
                __m256d ymm3 = _mm256_loadu2_m128d(&gate.mat_imag_[2], &gate.mat_imag_[2]); // m2 m3 m2 m3, imag

#pragma omp parallel for firstprivate(ymm0, ymm1, ymm2, ymm3)
                for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                    __m256d ymm4, ymm5, ymm6, ymm7, ymm8, ymm9;
                    auto inds = index(task_id, circuit_qubit_num, qubits, qubits_sorted);
                    if (qubits[0] == qubits_sorted[0]) { // ...q0.q1.
                        // v00 v10 v01 v11 ... v02 v12 v03 v13
                        ymm4 = _mm256_loadu_pd(&real[inds[2]]);
                        ymm5 = _mm256_loadu_pd(&imag[inds[2]]);
                    } else { // ...q1.q0.
                        // v00 v10 v02 v12 ... v01 v11 v03 v13
                        ymm4 = _mm256_loadu2_m128d(&real[inds[3]], &real[inds[2]]);
                        ymm5 = _mm256_loadu2_m128d(&imag[inds[3]], &imag[inds[2]]);
                    }
                    ymm4 = _mm256_permute4x64_pd(ymm4, 0b1101'1000); // v02 v03 v12 v13, real
                    ymm5 = _mm256_permute4x64_pd(ymm5, 0b1101'1000); // v02 v03 v12 v13, imag
                    COMPLEX_YMM_MUL(ymm0, ymm2, ymm4, ymm5, ymm6, ymm7);
                    COMPLEX_YMM_MUL(ymm1, ymm3, ymm4, ymm5, ymm8, ymm9);
                    ymm4 = _mm256_hadd_pd(ymm6, ymm8); // v02 v03 v12 v13, real
                    ymm5 = _mm256_hadd_pd(ymm7, ymm9); // v02 v03 v12 v13, imag
                    ymm4 = _mm256_permute4x64_pd(ymm4, 0b1101'1000); // v02 v12 v03 v13, real
                    ymm5 = _mm256_permute4x64_pd(ymm5, 0b1101'1000); // v02 v12 v03 v13, real
                    if (qubits[0] == qubits_sorted[0]) { // ...q0.q1.
                        // v00 v10 v01 v11 ... v02 v12 v03 v13
                        _mm256_storeu_pd(&real[inds[2]], ymm4);
                        _mm256_storeu_pd(&imag[inds[2]], ymm5);
                    } else { // ...q1.q0.
                        // v00 v10 v02 v12 ... v01 v11 v03 v13
                        _mm256_storeu2_m128d(&real[inds[3]], &real[inds[2]], ymm4);
                        _mm256_storeu2_m128d(&imag[inds[3]], &imag[inds[2]], ymm5);
                    }
                }
            } else if (qubits_sorted[1] < circuit_qubit_num - 2) { // ...q...q..
                constexpr uint64_t batch_size = 2;
                __m256d ymm0 = _mm256_loadu2_m128d(&gate.mat_real_[0], &gate.mat_real_[0]); // m0 m1 m0 m1, real
                __m256d ymm1 = _mm256_loadu2_m128d(&gate.mat_real_[2], &gate.mat_real_[2]); // m2 m3 m2 m3, real
                __m256d ymm2 = _mm256_loadu2_m128d(&gate.mat_imag_[0], &gate.mat_imag_[0]); // m0 m1 m0 m1, imag
                __m256d ymm3 = _mm256_loadu2_m128d(&gate.mat_imag_[2], &gate.mat_imag_[2]); // m2 m3 m2 m3, imag

#pragma omp parallel for firstprivate(ymm0, ymm1, ymm2, ymm3)
                for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                    auto inds = index(task_id, circuit_qubit_num, qubits, qubits_sorted);
                    __m256d ymm4 = _mm256_loadu2_m128d(&real[inds[3]], &real[inds[2]]); // v02 v12 v03 v13, real
                    __m256d ymm5 = _mm256_loadu2_m128d(&imag[inds[3]], &imag[inds[2]]); // v02 v12 v03 v13, imag
                    ymm4 = _mm256_permute4x64_pd(ymm4, 0b1101'1000); // v02 v03 v12 v13, real
                    ymm5 = _mm256_permute4x64_pd(ymm5, 0b1101'1000); // v02 v03 v12 v13, imag
                    __m256d ymm6, ymm7, ymm8, ymm9;
                    COMPLEX_YMM_MUL(ymm0, ymm2, ymm4, ymm5, ymm6, ymm7);
                    COMPLEX_YMM_MUL(ymm1, ymm3, ymm4, ymm5, ymm8, ymm9);
                    ymm4 = _mm256_hadd_pd(ymm6, ymm8); // v02 v03 v12 v13, real
                    ymm5 = _mm256_hadd_pd(ymm7, ymm9); // v02 v03 v12 v13, imag
                    ymm4 = _mm256_permute4x64_pd(ymm4, 0b1101'1000); // v02 v12 v03 v13, real
                    ymm5 = _mm256_permute4x64_pd(ymm5, 0b1101'1000); // v02 v12 v03 v13, real
                    _mm256_storeu2_m128d(&real[inds[3]], &real[inds[2]], ymm4);
                    _mm256_storeu2_m128d(&imag[inds[3]], &imag[inds[2]], ymm5);
                }
            }
        }
    }
}

#endif //SIM_BACK_MATRICKS_SIMULATOR_H
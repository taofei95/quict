//
// Created by Ci Lei on 2021-07-02.
//

#ifndef SIM_BACK_HYBRID_SIMULATOR_H
#define SIM_BACK_HYBRID_SIMULATOR_H

#include <cstdint>
#include <algorithm>
#include <vector>
#include <complex>
#include <pair>
#include <string>
#include <omp.h>
#include <immintrin.h>

#include "utility.h"
#include "monotonous_simulator.h"

namespace QuICT {
    template<typename Precision>
    class HybridSimulator {
    public:
        HybridSimulator() {
            using namespace std;
            if constexpr(!is_same_v<Precision, double> && !is_same_v<Precision, float>) {
                throw runtime_error("HybridSimulator only supports double/float precision.");
            }
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
        inline void run(
                uint64_t circuit_qubit_num,
                const std::vector<GateDescription<Precision>> &gate_desc_vec,
                const Precision *real,
                const Precision *imag
        );

        inline std::pair<Precision *, Precision *> separate_complex(
                uint64_t circuit_qubit_num,
                const complex <Precision> *c_arr
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

        template<typename Gate>
        inline void apply_gate(
                uint64_t circuit_qubit_num,
                const Gate &gate,
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
        auto pr = separate_complex(qubit_num, init_state);
        auto real = pr.first;
        auto imag = pr.second;
        run(qubit_num, gate_desc_vec, real, imag);
        combine_complex(qubit_num, real, imag, init_state);
        delete real;
        delete imag;
    }

    template<typename Precision>
    inline std::complex<Precision> *HybridSimulator<Precision>::run(
            uint64_t circuit_qubit_num,
            const std::vector<GateDescription<Precision>> &gate_desc_vec
    ) {
        auto len = 1ULL << qubit_num;
        auto real = new Precision[len];
        auto imag = new Precision[len];
        auto result = new std::complex<Precision>[len];
        run(qubit_num, gate_desc_vec, real, imag);
        combine_complex(qubit_num, real, imag, result);
        delete real;
        delete imag;
        return result;
    }

    template<typename Precision>
    inline void HybridSimulator<Precision>::run(
            uint64_t circuit_qubit_num,
            const std::vector<GateDescription<Precision>> &gate_desc_vec,
            const Precision *real,
            const Precision *imag
    ) {
        for (const auto &gate_desc:gate_desc_vec) {
            apply_gate(gate_desc, real, imag);
        }
    }

    template<typename Precision>
    inline std::pair<Precision *, Precision *>
    HybridSimulator<Precision>::separate_complex(uint64_t circuit_qubit_num, const complex <Precision> *c_arr) {
        auto len = 1ULL << qubit_num;
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
            std::complex<Precision> *res) {
        auto len = 1ULL << qubit_num;
        for (uint64_t i = 0; i < len; i += 4) {
            res[i] = {real[i], imag[i]};
            res[i + 1] = {real[i + 1], imag[i + 1]};
            res[i + 2] = {real[i + 2], imag[i + 2]};
            res[i + 3] = {real[i + 3], imag[i + 3]};
        }
        return res;
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
            apply_gate(circuit_qubit_num, gate, real, imag);
        } else if (gate_desc.qasm_name_ == "x") {
            auto gate = XGate<Precision>(gate_desc.affect_args_[0]);
            apply_gate(circuit_qubit_num, gate, real, imag);
        } else if (gate_desc.qasm_name_ == "crz") { // Two Bit
            auto gate = CrzGate<Precision>(gate_desc.affect_args_[0], gate_desc.affect_args_[1], gate_desc.parg_);
            apply_gate(circuit_qubit_num, gate, real, imag);
        } else { // Not Implemented
            throw std::runtime_error(std::string(__func__) + ": " + "Not implemented gate - " + gate_desc.qasm_name_);
        }
    }


    template<typename Precision>
    inline void HybridSimulator<Precision>::apply_gate(
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
                    // Shuffle combine
                    ymm1 = _mm256_shuffle_pd(ymm5, ymm6, 0b1010);
                    ymm2 = _mm256_shuffle_pd(ymm5, ymm6, 0b0101);
                    ymm3 = _mm256_shuffle_pd(ymm7, ymm8, 0b1010);
                    ymm4 = _mm256_shuffle_pd(ymm7, ymm8, 0b0101);
                    // Store
                    _mm256_storeu_pd(&real[ind_0[0]], ymm1);
                    _mm256_storeu_pd(&real[ind_0[0] + 4], ymm2);
                    _mm256_storeu_pd(&imag[ind_0[0]], ymm1);
                    _mm256_storeu_pd(&imag[ind_0[0] + 4], ymm2);
                }
            } else if (gate.targ_ == circuit_qubit_num - 2) {
                // After some permutations, this is the same with the previous one.
                constexpr uint64_t batch_size = 4;
                for (uint64_t task_id = 0; task_id < task_num; task_id += batch_size) {
                    auto ind_0 = index(task_id, circuit_qubit_num, gate.targ_);
                    auto cc = gate.sqrt2_inv.real();
                    // In memory swap
                    std::swap(real[ind_0[0] + 1], real[ind_0[0] + 2]);
                    std::swap(real[ind_0[0] + 5], real[ind_0[0] + 6]);
                    std::swap(imag[ind_0[0] + 1], imag[ind_0[0] + 2]);
                    std::swap(imag[ind_0[0] + 5], imag[ind_0[0] + 6]);
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
                    // Shuffle combine
                    ymm1 = _mm256_shuffle_pd(ymm5, ymm6, 0b1010);
                    ymm2 = _mm256_shuffle_pd(ymm5, ymm6, 0b0101);
                    ymm3 = _mm256_shuffle_pd(ymm7, ymm8, 0b1010);
                    ymm4 = _mm256_shuffle_pd(ymm7, ymm8, 0b0101);
                    // Store
                    _mm256_storeu_pd(&real[ind_0[0]], ymm1);
                    _mm256_storeu_pd(&real[ind_0[0] + 4], ymm2);
                    _mm256_storeu_pd(&imag[ind_0[0]], ymm1);
                    _mm256_storeu_pd(&imag[ind_0[0] + 4], ymm2);
                    // In memory swap
                    std::swap(real[ind_0[0] + 1], real[ind_0[0] + 2]);
                    std::swap(real[ind_0[0] + 5], real[ind_0[0] + 6]);
                    std::swap(imag[ind_0[0] + 1], imag[ind_0[0] + 2]);
                    std::swap(imag[ind_0[0] + 5], imag[ind_0[0] + 6]);
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

                    __m256d ymm5 = _mm256_mul_pd(ymm0, ymm1);          // c * real_row_0
                    __m256d ymm6 = _mm256_fmadd_pd(ymm0, ymm2, ymm5);  // c * real_row_0 + c * real_row_1
                    __m256d ymm7 = _mm256_fnmadd_pd(ymm0, ymm2, ymm5); // c * real_row_0 - c * real_row_1

                    __m256d ymm8 = _mm256_mul_pd(ymm0, ymm3);          // c * imag_row_0
                    __m256d ymm9 = _mm256_fmadd_pd(ymm0, ymm4, ymm8);  // c * imag_row_0 + c * imag_row_1
                    __m256d ymm10 = _mm256_fnmadd_pd(ymm0, ymm4, ymm8);// c * imag_row_0 - c * imag_row_1

                    _mm256_storeu_pd(&real[ind_0[0]], ymm6);
                    _mm256_storeu_pd(&real[ind_0[1]], ymm7);
                    _mm256_storeu_pd(&imag[ind_0[0]], ymm9);
                    _mm256_storeu_pd(&imag[ind_0[1]], ymm10);
                }
            }
        }
    }
}

#endif //SIM_BACK_HYBRID_SIMULATOR_H

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
#include <random>

#include "gate.h"
#include "utility.h"

namespace QuICT {
    template<typename Precision>
    class MaTricksSimulator {
    protected:
        std::string name_;
        const Details::SysConfig sysconfig_;
        std::default_random_engine random_gen;
        std::uniform_real_distribution<double> random_dist;

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

            random_gen.seed(std::chrono::system_clock::now().time_since_epoch().count());
            random_dist =  std::uniform_real_distribution<double>(0.0,1.0);
        }

        inline const std::string &name() {
            return name_;
        }

        inline void apply_gate(
                uint64_t q_state_bit_num,
                const GateDescription<Precision> &gate_desc,
                Precision *real,
                Precision *imag
        );

        inline void run(
                uint64_t q_state_bit_num,
                const std::vector<GateDescription<Precision>> &gate_desc_vec,
                const std::complex<Precision> *init_state
        );

        inline std::complex<Precision> *run(
                uint64_t q_state_bit_num,
                const std::vector<GateDescription<Precision>> &gate_desc_vec
        );

    private:

        inline void run(
                uint64_t q_state_bit_num,
                const std::vector<GateDescription<Precision>> &gate_desc_vec,
                Precision *real,
                Precision *imag
        );

        template<uint64_t N, template<uint64_t, typename> class Gate>
        inline void apply_diag_n_gate(
                uint64_t q_state_bit_num,
                const Gate<N, Precision> &gate,
                Precision *real,
                Precision *imag
        );

        template<template<typename> class Gate>
        inline void apply_ctrl_diag_gate(
                uint64_t q_state_bit_num,
                const Gate<Precision> &gate,
                Precision *real,
                Precision *imag
        );

        template<uint64_t N, template<uint64_t, typename> class Gate>
        inline void apply_unitary_n_gate(
                uint64_t q_state_bit_num,
                const Gate<N, Precision> &gate,
                Precision *real,
                Precision *imag
        );

        template<template<typename> class Gate>
        inline void apply_ctrl_unitary_gate(
                uint64_t q_state_bit_num,
                const Gate<Precision> &gate,
                Precision *real,
                Precision *imag
        );

        template<template<typename> class Gate>
        inline void apply_h_gate(
                uint64_t q_state_bit_num,
                const Gate<Precision> &gate,
                Precision *real,
                Precision *imag
        );

        template<template<typename> class Gate>
        inline void apply_x_gate(
                uint64_t q_state_bit_num,
                const Gate<Precision> &gate,
                Precision *real,
                Precision *imag
        );

        inline void apply_measure_gate(
                uint64_t q_state_bit_num,
                const MeasureGate &gate,
                Precision *real,
                Precision *imag
        );
    };

    template<typename Precision>
    void MaTricksSimulator<Precision>::run(
            uint64_t q_state_bit_num,
            const std::vector<GateDescription<Precision>> &gate_desc_vec,
            const std::complex<Precision> *init_state
    ) {
        auto pr = separate_complex(q_state_bit_num, init_state);
        auto real = pr.first;
        auto imag = pr.second;
        run(q_state_bit_num, gate_desc_vec, real, imag);
        combine_complex(q_state_bit_num, real, imag, init_state);

        delete[] real;
        delete[] imag;
    }

    template<typename Precision>
    std::complex<Precision> *MaTricksSimulator<Precision>::run(
            uint64_t q_state_bit_num,
            const std::vector<GateDescription<Precision>> &gate_desc_vec
    ) {
        auto len = 1ULL << q_state_bit_num;
        auto real = new Precision[len];
        auto imag = new Precision[len];
        auto result = new std::complex<Precision>[len];
        std::fill(real, real + len, 0);
        std::fill(imag, imag + len, 0);
        real[0] = 1.0;

        run(q_state_bit_num, gate_desc_vec, real, imag);
        combine_complex(q_state_bit_num, real, imag, result);
        delete[] real;
        delete[] imag;
        return result;
    }

    template<typename Precision>
    void MaTricksSimulator<Precision>::run(
            uint64_t q_state_bit_num,
            const std::vector<GateDescription<Precision>>

            &gate_desc_vec,
            Precision *real,
            Precision *imag
    ) {
        for (const auto &gate_desc: gate_desc_vec) {
            apply_gate(q_state_bit_num, gate_desc, real, imag);
        }
    }

    template<typename Precision>
    void MaTricksSimulator<Precision>::apply_gate(
            uint64_t q_state_bit_num,
            const GateDescription<Precision> &gate_desc,
            Precision *real,
            Precision *imag
    ) {
        auto search = dispatcher.find(gate_desc.gate_name_);
        if (search == dispatcher.end()) {
            throw std::runtime_error(std::string(__func__) + ": " + "Not implemented gate - " + gate_desc.gate_name_);
        } else {
            gate_category gate_category = search->second;
            switch (gate_category) {
                case gate_category::special_x: {
                    auto gate = XGate<Precision>(gate_desc.affect_args_[0]);
                    apply_x_gate(q_state_bit_num, gate, real, imag);
                    break;
                }
                case gate_category::special_h: {
                    auto gate = HGate<Precision>(gate_desc.affect_args_[0]);
                    apply_h_gate(q_state_bit_num, gate, real, imag);
                    break;
                }
                case gate_category::diag_1: {
                    auto diag_1_gate = DiagonalGateN<1, Precision>(gate_desc.affect_args_[0], gate_desc.data_ptr_);
                    apply_diag_n_gate(q_state_bit_num, diag_1_gate, real, imag);
                    break;
                }
                case gate_category::diag_2: {
                    auto diag_2_gate = DiagonalGateN<2, Precision>(gate_desc.affect_args_, gate_desc.data_ptr_);
                    apply_diag_n_gate(q_state_bit_num, diag_2_gate, real, imag);
                    break;
                }
                case gate_category::ctrl_diag: {
                    auto ctrl_diag_gate = ControlledDiagonalGate<Precision>(
                            gate_desc.affect_args_[0],
                            gate_desc.affect_args_[1],
                            gate_desc.data_ptr_
                    );
                    apply_ctrl_diag_gate(q_state_bit_num, ctrl_diag_gate, real, imag);
                    break;
                }
                case gate_category::unitary_1: {
                    auto unitary_1_gate = UnitaryGateN<1, Precision>(gate_desc.affect_args_[0], gate_desc.data_ptr_);
                    apply_unitary_n_gate(q_state_bit_num, unitary_1_gate, real, imag);
                    break;
                }
                case gate_category::unitary_2: {
                    auto unitary_2_gate = UnitaryGateN<2, Precision>(gate_desc.affect_args_, gate_desc.data_ptr_);
                    apply_unitary_n_gate(q_state_bit_num, unitary_2_gate, real, imag);
                    break;
                }
                case gate_category::ctrl_unitary: {
                    auto ctrl_unitary_gate = ControlledUnitaryGate<Precision>(
                            gate_desc.affect_args_[0],
                            gate_desc.affect_args_[1],
                            gate_desc.data_ptr_
                    );
                    apply_ctrl_unitary_gate(q_state_bit_num, ctrl_unitary_gate, real, imag);
                    break;
                }
                case gate_category::measure: {
                    auto measure_gate = MeasureGate(gate_desc.affect_args_[0]);
                    apply_measure_gate(q_state_bit_num, measure_gate, real, imag);
                    break;
                }
                default: {
                    throw std::runtime_error(
                            std::string(__func__) + ": " + "Not implemented gate - " + gate_desc.gate_name_);
                }
            };
        }
    }

    // Original source file has complex structure for IDE static analysis.
    // Separate it into `.tcc` implementations.

    //**********************************************************************
    // Special simple gates
    //**********************************************************************

    //**********************************************************************
    // Special matrix pattern gates
    //**********************************************************************
}

#include "avx_impl/avx_x_gate.tcc"
#include "avx_impl/avx_ctrl_diag_gate.tcc"
#include "avx_impl/avx_diag_n_gate.tcc"
#include "avx_impl/avx_unitary_n_gate.tcc"
#include "avx_impl/avx_ctrl_unitary_gate.tcc"
#include "avx_impl/avx_h_gate.tcc"
#include "avx_impl/avx_measure_gate.tcc"

#endif //SIM_BACK_MATRICKS_SIMULATOR_H
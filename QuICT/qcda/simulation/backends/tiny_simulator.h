//
// Created by Ci Lei on 2021-10-05.
//

#ifndef SIM_BACK_TINY_SIMULATOR_H
#define SIM_BACK_TINY_SIMULATOR_H

#include <vector>
#include <complex>
#include <string>

#include "utility.h"
#include "gate.h"

namespace QuICT {
    // Support for small q-state
    template<typename Precision>
    class TinySimulator {
    protected:
        std::string name_;
    public:
        TinySimulator() {
            using namespace std;
            if constexpr(!is_same_v<Precision, double> && !is_same_v<Precision, float>) {
                throw runtime_error("MaTricksSimulator only supports double/float precision.");
            }
            name_ = "TinySimulator";
            if constexpr(std::is_same_v<Precision, double>) {
                name_ += "[double]";
            } else if (std::is_same_v<Precision, float>) {
                name_ += " [float]";
            }
        }

        inline const std::string &name() {
            return name_;
        };

        inline std::complex<Precision> *run(
                uint64_t q_state_bit_num,
                const std::vector<GateDescription<Precision>> &gate_desc_vec
        );

        inline void run(
                uint64_t q_state_bit_num,
                const std::vector<GateDescription<Precision>> &gate_desc_vec,
                Precision *real,
                Precision *imag
        );

        inline void apply_gate(
                uint64_t q_state_bit_num,
                const GateDescription<Precision> &gate_desc,
                Precision *real,
                Precision *imag
        );

    private:
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
        void apply_unitary_n_gate(
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
    };

    template<typename Precision>
    std::complex<Precision> *TinySimulator<Precision>::run(
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
    void TinySimulator<Precision>::run(
            uint64_t q_state_bit_num,
            const std::vector<GateDescription<Precision>> &gate_desc_vec,
            Precision *real,
            Precision *imag
    ) {
        for (const auto &gate_desc: gate_desc_vec) {
            apply_gate(q_state_bit_num, gate_desc, real, imag);
        }
    }

    template<typename Precision>
    void TinySimulator<Precision>::apply_gate(
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
                    throw std::runtime_error(
                            std::string(__func__) + ": " + "Not implemented gate - " + gate_desc.gate_name_);
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
                default: {
                    throw std::runtime_error(
                            std::string(__func__) + ": " + "Not implemented gate - " + gate_desc.gate_name_);
                }
            };
        }
    }

    template<typename Precision>
    template<uint64_t N, template<uint64_t, typename> class Gate>
    void TinySimulator<Precision>::apply_diag_n_gate(
            uint64_t q_state_bit_num,
            const Gate<N, Precision> &gate,
            Precision *real,
            Precision *imag
    ) {
        if constexpr(N == 1) {
            uint64_t task_num = 1ULL << (q_state_bit_num - 1);
            for (uint64_t task_id = 0; task_id < task_num; task_id += 1) {
                auto inds = index(task_id, q_state_bit_num, gate.targ_);
                for (int i = 0; i < 2; ++i) {
                    auto res_r = real[inds[i]] * gate.diagonal_real_[i] - imag[inds[i]] * gate.diagonal_imag_[i];
                    auto res_i = real[inds[i]] * gate.diagonal_imag_[i] + imag[inds[i]] * gate.diagonal_real_[i];
                    real[i] = res_r;
                    imag[i] = res_i;
                }
            }
        } else {
            throw std::runtime_error(
                    std::string(__func__) + ": " + "Not implemented for diag gate >= 2");
        }
    }

    template<typename Precision>
    template<template<typename> class Gate>
    void TinySimulator<Precision>::apply_ctrl_diag_gate(
            uint64_t q_state_bit_num,
            const Gate<Precision> &gate,
            Precision *real,
            Precision *imag
    ) {
        // TODO: Finish this
    }

    template<typename Precision>
    template<uint64_t N, template<uint64_t, typename> class Gate>
    void TinySimulator<Precision>::apply_unitary_n_gate(
            uint64_t q_state_bit_num,
            const Gate<N, Precision> &gate,
            Precision *real,
            Precision *imag
    ) {
        // TODO: Finish this
    }

    template<typename Precision>
    template<template<typename> class Gate>
    inline void TinySimulator<Precision>::apply_ctrl_unitary_gate(
            uint64_t q_state_bit_num,
            const Gate<Precision> &gate,
            Precision *real,
            Precision *imag
    ) {
        // TODO: Finish this
    }

    template<typename Precision>
    template<template<typename> class Gate>
    inline void TinySimulator<Precision>::apply_h_gate(
            uint64_t q_state_bit_num,
            const Gate<Precision> &gate,
            Precision *real,
            Precision *imag
    ) {
        uint64_t task_num = 1ULL << (q_state_bit_num - 1);
        for (uint64_t task_id = 0; task_id < task_num; task_id += 1) {
            auto inds = index(task_id, q_state_bit_num, gate.targ_);
            Precision sqrt2_inv = gate.sqrt2_inv.real();
            Precision r0 = real[inds[0]], r1 = real[inds[1]];
            Precision i0 = imag[inds[0]], i1 = imag[inds[1]];
            real[inds[0]] = (r0 + r1) * sqrt2_inv;
            imag[inds[0]] = (i0 + i1) * sqrt2_inv;
            real[inds[1]] = (r0 - r1) * sqrt2_inv;
            imag[inds[1]] = (i0 - i1) * sqrt2_inv;
        }
    }

    template<typename Precision>
    template<template<typename> class Gate>
    inline void TinySimulator<Precision>::apply_x_gate(
            uint64_t q_state_bit_num,
            const Gate<Precision> &gate,
            Precision *real,
            Precision *imag
    ) {
        // TODO: Finish this
    }
}

#endif //SIM_BACK_TINY_SIMULATOR_H

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

#include "utility.h"
#include "monotonous_simulator.h"

namespace QuICT {
    template<typename Precision>
    class HybridSimulator {
    public:
        inline void run(
                uint64_t qubit_num,
                const std::vector<GateDescription<Precision>> &gate_desc_vec,
                const std::complex<Precision> *init_state
        );

    private:
        inline void run(
                uint64_t qubit_num,
                const std::vector<GateDescription<Precision>> &gate_desc_vec,
                const Precision *real,
                const Precision *imag
        );

        inline std::pair<Precision *, Precision *> separate_complex(
                uint64_t qubit_num,
                const complex <Precision> *c_arr
        );

        inline void combine_complex(
                uint64_t qubit_num,
                const Precision *real,
                const Precision *imag,
                std::complex<Precision> *res
        );

        inline void apply_gate(
                const GateDescription<Precision> &gate_desc,
                Precision *real,
                Precision *imag
        );

        template<typename Gate>
        inline void apply_gate(
                const Gate &gate,
                Precision *real,
                Precision *imag
        );
    };

    template<typename Precision>
    inline void HybridSimulator<Precision>::run(
            uint64_t qubit_num,
            const std::vector<GateDescription<Precision>> &gate_desc_vec,
            const std::complex<Precision> *init_state
    ) {
        auto pr = separate_complex(qubit_num, init_state);
        auto real = pr.first;
        auto imag = pr.second;
        run(qubit_num, gate_desc_vec, real, imag);
        combine_complex(qubit_num, real, imag, init_state);
        delete[] real;
        delete[] imag;
    }

    template<typename Precision>
    inline void HybridSimulator<Precision>::run(
            uint64_t qubit_num,
            const std::vector<GateDescription<Precision>> &gate_desc_vec,
            const Precision *real,
            const Precision *imag
    ) {
        return nullptr;
    }

    template<typename Precision>
    inline std::pair<Precision *, Precision *>
    HybridSimulator<Precision>::separate_complex(uint64_t qubit_num, const complex <Precision> *c_arr) {
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
            uint64_t qubit_num,
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
            const GateDescription<Precision> &gate_desc,
            Precision *real,
            Precision *imag
    ) {
        if (gate_desc.qasm_name_ == "h") { // Single Bit
            auto gate = HGate<Precision>(gate_desc.affect_args_[0]);
            apply_gate(gate, real, imag);
        } else if (gate_desc.qasm_name_ == "x") {
            auto gate = XGate<Precision>(gate_desc.affect_args_[0]);
            apply_gate(gate, real, imag);
        } else if (gate_desc.qasm_name_ == "crz") { // Two Bit
            auto gate = CrzGate<Precision>(gate_desc.affect_args_[0], gate_desc.affect_args_[1], gate_desc.parg_);
            apply_gate(gate, real, imag);
        } else { // Not Implemented
            throw std::runtime_error(std::string(__func__) + ": " + "Not implemented gate - " + gate_desc.qasm_name_);
        }
    }


    template<typename Precision>
    template<typename Gate>
    inline void HybridSimulator<Precision>::apply_gate(
            const Gate &gate,
            Precision *real,
            Precision *imag
    ) {

    }
}

#endif //SIM_BACK_HYBRID_SIMULATOR_H

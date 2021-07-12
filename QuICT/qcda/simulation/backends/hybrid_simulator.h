//
// Created by Ci Lei on 2021-07-02.
//

#ifndef SIM_BACK_HYBRID_SIMULATOR_H
#define SIM_BACK_HYBRID_SIMULATOR_H

#include <cstdint>
#include <algorithm>
#include <vector>
#include <complex>

#include "utility.h"
#include "monotune_simulator.h"

namespace QuICT {
    template<typename precision_t>
    class HybridSimulator {
    public:
        void run(
                uint64_t qubit_num,
                const std::vector<GateDescription<precision_t>> &gate_desc_vec
        );

        void run(
                uint64_t qubit_num,
                const std::vector<GateDescription<precision_t>> &gate_desc_vec,
                std::complex<precision_t> *init_state
        );

        template<class Gate>
        void apply_gate(const GateDescription<precision_t> &gate_desc);

        template<class Gate>

    };

    template<typename precision_t>
    void HybridSimulator<precision_t>::run(
            uint64_t qubit_num,
            const std::vector<GateDescription<precision_t>> &gate_desc_vec
    ) {

    }

    template<typename precision_t>
    void HybridSimulator<precision_t>::run(
            uint64_t qubit_num,
            const std::vector<GateDescription<precision_t>> &gate_desc_vec,
            std::complex<precision_t> *init_state
    ) {

    }

    template<typename precision_t>
    template<class Gate>
    void HybridSimulator<precision_t>::apply_gate(const GateDescription<precision_t> &gate_desc) {

    }

}

#endif //SIM_BACK_HYBRID_SIMULATOR_H

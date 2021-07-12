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
    template<typename Precision>
    class HybridSimulator {
    public:
        void run(
                uint64_t qubit_num,
                const std::vector<GateDescription<Precision>> &gate_desc_vec
        );

        void run(
                uint64_t qubit_num,
                const std::vector<GateDescription<Precision>> &gate_desc_vec,
                std::complex<Precision> *init_state
        );

        template<class Gate>
        void apply_gate(const GateDescription<Precision> &gate_desc);

        template<class Gate>
    };

    template<typename Precision>
    void HybridSimulator<Precision>::run(
            uint64_t qubit_num,
            const std::vector<GateDescription<Precision>> &gate_desc_vec
    ) {

    }

    template<typename Precision>
    void HybridSimulator<Precision>::run(
            uint64_t qubit_num,
            const std::vector<GateDescription<Precision>> &gate_desc_vec,
            std::complex<Precision> *init_state
    ) {

    }

    template<typename Precision>
    template<class Gate>
    void HybridSimulator<Precision>::apply_gate(const GateDescription<Precision> &gate_desc) {

    }

}

#endif //SIM_BACK_HYBRID_SIMULATOR_H

//
// Created by LinHe on 2021-10-05.
//

#ifndef SIM_BACK_TINY_SIMULATOR_H
#define SIM_BACK_TINY_SIMULATOR_H

#include <vector>

#include "utility.h"

namespace QuICT {
    // Support for small q-state
    template<typename Precision>
    class TinySimulator {
    public:
        void run(
                uint64_t q_state_bit_num,
                const std::vector<GateDescription<Precision>> &gate_desc_vec
        ) {

        }
    };
}

#endif //SIM_BACK_TINY_SIMULATOR_H

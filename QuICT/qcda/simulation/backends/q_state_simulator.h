//
// Created by LinHe on 2021-09-24.
//

#ifndef SIM_BACK_Q_STATE_SIMULATOR_H
#define SIM_BACK_Q_STATE_SIMULATOR_H

#include <cstdint>
#include <vector>

#include "utility.h"
#include "gate.h"
#include "matricks_simulator.h"

namespace QuICT {
    class QState {
    public:
        uint64_t id;

    };


    template<typename Precision>
    class QStateSimulator {
    protected:
        TinySimulator <Precision> tiny_sim_;
        MatricksSimulator <Precision> matricks_sim_;

    public:
        void run(
                uint64_t q_state_bit_num,
                const std::vector<GateDescription<Precision>> &gate_desc_vec
        ) {

        }
    };
}


#endif //SIM_BACK_Q_STATE_SIMULATOR_H

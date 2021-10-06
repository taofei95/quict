//
// Created by Ci Lei on 2021-09-24.
//

#ifndef SIM_BACK_Q_STATE_SIMULATOR_H
#define SIM_BACK_Q_STATE_SIMULATOR_H

#include <cstdint>
#include <vector>

#include "utility.h"
#include "gate.h"
#include "tiny_simulator.h"
#include "matricks_simulator.h"

namespace QuICT {
    class QState {
    public:
        uint64_t id;

    };


    template<typename Precision>
    class QStateSimulator {
    protected:
        TinySimulator<Precision> tiny_sim_;
        MatricksSimulator <Precision> matricks_sim_;

    public:
        inline void apply_gate(
                QState q_state,
                const GateDescription<Precision> &gate_desc
        );
    };

    template<typename Precision>
    void QStateSimulator<Precision>::apply_gate(
            QState q_state,
            const GateDescription<Precision> &gate_desc
    ) {
        // TODO: Finish this
    }
}


#endif //SIM_BACK_Q_STATE_SIMULATOR_H

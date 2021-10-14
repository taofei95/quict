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
    template<typename Precision>
    class QState {
    public:
        uint64_t id;
        std::vector<uint64_t> qubits;
        Precision *real;
        Precision *imag;

        static QState<Precision> merge(const QState<Precision> &a, const QState<Precision> &b) {

        }
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
        if (q_state.qubits.size() <= 4) {
            tiny_sim_.apply_gate(q_state.qubits.size(), gate_desc, q_state.real, q_state.imag);
        } else {
            matricks_sim_.apply_gate(q_state.qubits.size(), gate_desc, q_state.real, q_state.imag);
        }
    }
}


#endif //SIM_BACK_Q_STATE_SIMULATOR_H

//
// Created by Ci Lei on 2021-09-24.
//

#ifndef SIM_BACK_Q_STATE_SIMULATOR_H
#define SIM_BACK_Q_STATE_SIMULATOR_H

#include <string>
#include <cstdint>
#include <vector>
#include <map>

#include "utility.h"
#include "q_state.h"
#include "q_state_set.h"
#include "gate.h"
#include "tiny_simulator.h"
#include "matricks_simulator.h"

namespace QuICT {


    template<typename Precision>
    class QStateSimulator {
    protected:
        std::string name_;
        inline static TinySimulator<Precision> tiny_sim_;
        inline static MaTricksSimulator<Precision> matricks_sim_;

    public:
        QStateSimulator() {
            using namespace std;
            if constexpr(!is_same_v<Precision, double> && !is_same_v<Precision, float>) {
                throw runtime_error("MaTricksSimulator only supports double/float precision.");
            }
            name_ = "QStateSimulator";
            if constexpr(std::is_same_v<Precision, double>) {
                name_ += "[double]";
            } else if (std::is_same_v<Precision, float>) {
                name_ += " [float]";
            }
        }

        inline void apply_gate(
                const QState<Precision> &q_state,
                const GateDescription<Precision> &gate_desc
        );
    };

    template<typename Precision>
    void QStateSimulator<Precision>::apply_gate(
            const QState<Precision> &q_state,
            const GateDescription<Precision> &gate_desc
    ) {
        auto desc_cpy = gate_desc;
        for (auto &it: desc_cpy.affect_args_) {
            it = q_state.qubit_mapping_.at(it);
        }
        if (q_state.qubit_num_ <= 4) {
            tiny_sim_.apply_gate(q_state.qubit_num_, desc_cpy, q_state.real_, q_state.imag_);
        } else {
            matricks_sim_.apply_gate(q_state.qubit_num_, desc_cpy, q_state.real_, q_state.imag_);
        }
    }
}


#endif //SIM_BACK_Q_STATE_SIMULATOR_H

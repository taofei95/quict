//
// Created by LinHe on 2021-09-24.
//

#ifndef SIM_BACK_CIRCUIT_SIMULATOR_H
#define SIM_BACK_CIRCUIT_SIMULATOR_H

#include <string>
#include <vector>
#include <map>
#include <complex>

#include "utility.h"
#include "gate.h"
#include "q_state.h"
#include "q_state_set.h"
#include "q_state_simulator.h"
#include "matricks_simulator.h"

namespace QuICT {
    template<typename Precision>
    class CircuitSimulator {
    public:
        CircuitSimulator(uint64_t qubit_num)
                : qubit_num_(qubit_num), q_state_set_(qubit_num) {
            // name
            using namespace std;
            if constexpr(!is_same_v<Precision, double> && !is_same_v<Precision, float>) {
                throw runtime_error("MaTricksSimulator only supports double/float precision.");
            }
            name_ = "CircuitSimulator";
            if constexpr(std::is_same_v<Precision, double>) {
                name_ += "[double]";
            } else if (std::is_same_v<Precision, float>) {
                name_ += " [float]";
            }
        }

        const std::string &name() {
            return name_;
        }

        inline std::complex<Precision> *run(const std::vector<GateDescription<Precision>> &gate_desc_vec);

    private:
        std::string name_;
        uint64_t qubit_num_;
        QStateSet<Precision> q_state_set_;
        QStateSimulator<Precision> q_state_simulator_;

        inline void apply_gate(const GateDescription<Precision> &gate_desc);
    };

    template<typename Precision>
    std::complex<Precision> *CircuitSimulator<Precision>::run(
            const std::vector<GateDescription<Precision>> &gate_desc_vec
    ) {
        for (const auto &gate_desc: gate_desc_vec) {
            apply_gate(gate_desc);
        }
        auto state = this->q_state_set_.merge_all();
        state->mapping_back();
        auto res = new std::complex<Precision>[1ULL << state->qubit_num_];
        combine_complex(state->qubit_num_, state->real_, state->imag_, res);
        return res;
    }

    template<typename Precision>
    void CircuitSimulator<Precision>::apply_gate(const GateDescription<Precision> &gate_desc) {
        auto state = q_state_set_.get_q_state(gate_desc.affect_args_[0]);
        for (uint64_t i = 1; i < gate_desc.affect_args_.size(); ++i) {
            state = q_state_set_.merge_q_state(gate_desc.affect_args_[0], gate_desc.affect_args_[i]);
        }
        q_state_simulator_.apply_gate(*state, gate_desc);
    }
};


#endif //SIM_BACK_CIRCUIT_SIMULATOR_H

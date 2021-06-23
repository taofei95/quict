//
// Created by Ci Lei on 2021-06-23.
//

#ifndef SIMULATION_BACKENDS_SIMULATOR_H
#define SIMULATION_BACKENDS_SIMULATOR_H

#include <cstdint>
#include "state_vector.h"
#include "gate.h"
#include "utility.h"

namespace QuICT {

    template<typename precision_t>
    class Simulator {
    public:
        template<class gate_t>
        inline void apply_gate(const gate_t &gate) {
            uint64_t task_num = 1ULL << (qubit_num_ - Gate::gate_qubit_num<gate_t>::value);
            for (uint64_t task_id = 0; task_id < task_num; ++task_id) {
                apply_gate_single_task(task_id, gate);
            }
        }

    private:
        // One task one run.
        inline void apply_gate_single_task(
                const uint64_t task_id,
                const Gate::HGate<precision_t> &gate
        ) {
            auto ind = index(task_id, qubit_num_, gate.targ_);
            auto tmp_arr_1 = marray_t<precision_t, 2>();
            tmp_arr_1[0] = state_vector_[ind[0]];
            tmp_arr_1[1] = state_vector_[ind[1]];
            state_vector_[ind[0]] = gate.sqrt2_inv * tmp_arr_1[0] + gate.sqrt2_inv * tmp_arr_1[1];
            state_vector_[ind[1]] = gate.sqrt2_inv * tmp_arr_1[0] - gate.sqrt2_inv * tmp_arr_1[1];
        }

        inline void apply_gate_single_task(
                const uint64_t task_id,
                const Gate::CrzGate<precision_t> &gate
        ) {

        }

    protected:
        StateVector<precision_t> state_vector_;
        uint64_t qubit_num_;
    };
}

#endif //SIMULATION_BACKENDS_SIMULATOR_H

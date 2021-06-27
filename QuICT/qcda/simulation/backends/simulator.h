//
// Created by Ci Lei on 2021-06-23.
//

#ifndef SIMULATION_BACKENDS_SIMULATOR_H
#define SIMULATION_BACKENDS_SIMULATOR_H

#include <cstdint>
#include <algorithm>
#include <vector>
#include <complex>
#include "gate.h"
#include "utility.h"

namespace QuICT {

    enum SimulatorMode {
        single,
        batch,
        avx,
        fma
    };


    template<typename precision_t, SimulatorMode sim_mode = SimulatorMode::single>
    class Simulator {
    public:
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        // Data Access Helper
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        void append_gate(
                std::string qasm_name,
                uint64_t targ,
                uint64_t carg,
                precision_t parg,
                std::vector<uint64_t> affect_args,
                mat_entry_t <precision_t> data_ptr
        ) {
            gate_vec_.emplace_back(GateBridgeEntry<precision_t>(
                    qasm_name, targ, carg, parg, affect_args, data_ptr
            ));
        }

        void clear_gate() {
            gate_vec_.clear();
        }

        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        // Run Simulation
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        void run(uint64_t qubit_num, std::complex<precision_t> *init_state) {
            qubit_num_ = qubit_num;
            state_vector_ = init_state;
            for (const auto &gate_desc: gate_vec_) {
                if (gate_desc.qasm_name_ == "h") {
                    auto gate = build_gate<Gate::HGate<precision_t>>(gate_desc);
                    apply_gate(gate);
                } else if (gate_desc.qasm_name_ == "crz") {
                    auto gate = build_gate<Gate::CrzGate<precision_t>>(gate_desc);
                    apply_gate(gate);
                } else {
                    throw std::runtime_error("Not implemented gate: " + gate_desc.qasm_name_);
                }
            }

            qubit_num_ = 0;
            state_vector_ = nullptr;
        }

    private:
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        // Private Simulation Methods
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        template<typename gate_t>
        gate_t build_gate(const GateBridgeEntry <precision_t> &gate_desc) {
            using namespace Gate;
            using namespace std;
            if constexpr(is_same<HGate<precision_t>, gate_t>::value) {
                // H gate
                return HGate<precision_t>(gate_desc.targ_);
            } else if constexpr(is_same<CrzGate<precision_t>, gate_t>::value) {
                // Crz gate
                return CrzGate<precision_t>(gate_desc.carg_, gate_desc.targ_, gate_desc.parg_);
            } else {
                throw runtime_error("Cannot build gate for " + gate_desc.qasm_name_);
            }
        }

        template<class gate_t>
        inline void apply_gate(const gate_t &gate) {
            if constexpr(sim_mode == SimulatorMode::single) {
                uint64_t task_num = 1ULL << (qubit_num_ - Gate::gate_qubit_num<gate_t>::value);
                for (uint64_t task_id = 0; task_id < task_num; ++task_id) {
                    apply_gate_single_task(task_id, gate);
                }
            } else if constexpr(sim_mode == SimulatorMode::batch) {

            } else if constexpr(sim_mode == SimulatorMode::avx) {

            } else if constexpr(sim_mode == SimulatorMode::fma) {

            }
        }

        // One task per run.
        inline void apply_gate_single_task(
                const uint64_t task_id,
                const Gate::HGate <precision_t> &gate
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
                const Gate::CrzGate <precision_t> &gate
        ) {
            uarray_t<2> qubits = {gate.carg_, gate.targ_};
            uarray_t<2> qubits_sorted = {gate.carg_, gate.targ_};
            std::sort(qubits_sorted.begin(), qubits_sorted.end());
            auto ind = index(task_id, qubit_num_, qubits, qubits_sorted);
            state_vector_[ind[2]] = gate.diagonal_[0] * state_vector_[ind[2]];
            state_vector_[ind[3]] = gate.diagonal_[1] * state_vector_[ind[3]];
        }

    protected:
        std::complex<precision_t> *state_vector_ = nullptr;
        uint64_t qubit_num_ = 0;
        std::vector<GateBridgeEntry < precision_t>> gate_vec_;
    };
}

#endif //SIMULATION_BACKENDS_SIMULATOR_H

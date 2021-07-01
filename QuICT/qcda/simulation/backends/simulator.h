//
// Created by Ci Lei on 2021-06-23.
//

#ifndef SIMULATION_BACKENDS_SIMULATOR_H
#define SIMULATION_BACKENDS_SIMULATOR_H

#include <cstdint>
#include <algorithm>
#include <vector>
#include <complex>
#include "utility.h"
#include "gate.h"

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
        // Constructor & Deconstructor
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        ~Simulator() {
            for (auto &gate_desc:gate_vec_) {
                delete[] gate_desc.data_ptr_;
            }
        }


        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        // Data Access Helper
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        void append_gate(
                std::string qasm_name,
                std::vector<uint64_t> affect_args,
                precision_t parg,
                mat_entry_t <precision_t> *data_ptr
        ) {
            gate_vec_.emplace_back(GateBridgeEntry<precision_t>(
                    qasm_name, affect_args, parg, data_ptr
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
                    apply_gate<Gate::HGate<precision_t>>(gate_desc);
                } else if (gate_desc.qasm_name_ == "crz") {
                    apply_gate<Gate::CrzGate<precision_t>>(gate_desc);
                } else {
                    throw std::runtime_error(
                            std::string(__func__) + ": " + "Not implemented gate: " + gate_desc.qasm_name_);
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
        inline void apply_gate(const GateBridgeEntry <precision_t> &gate_desc) {
            using namespace Gate;
            using namespace std;
            if constexpr(is_same<HGate<precision_t>, gate_t>::value) {
                // H gate
                auto gate = HGate<precision_t>(gate_desc.affect_args_[0]);
                apply_gate(gate);
            } else if constexpr(is_same<CrzGate<precision_t>, gate_t>::value) {
                // Crz gate
                auto gate = CrzGate<precision_t>(
                        gate_desc.affect_args_[0],
                        gate_desc.affect_args_[1],
                        gate_desc.parg_);
                apply_gate(gate);
            } else if constexpr(gate_has_mat_repr<gate_t>::value) {
                const auto gate_qubit_num = gate_desc.affect_args_.size();
                switch (gate_qubit_num) {
                    case 1: {
                        auto gate = UnitaryGateN<1, precision_t>(gate_desc.affect_args_.begin(),
                                                                 gate_desc.data_ptr_);
                        apply_gate(gate);
                        break;
                    }
                    case 2: {
                        auto gate = UnitaryGateN<2, precision_t>(gate_desc.affect_args_.begin(),
                                                                 gate_desc.data_ptr_);
                        apply_gate(gate);
                        break;
                    }
                    case 3: {
                        auto gate = UnitaryGateN<3, precision_t>(gate_desc.affect_args_.begin(),
                                                                 gate_desc.data_ptr_);
                        apply_gate(gate);
                        break;
                    }
                    default: {
                        throw std::runtime_error(
                                std::string(__func__) + ": " + "Not implemented for gate larger than 3");
                    }
                }
            } else {
                throw runtime_error("Cannot apply gate for " + gate_desc.qasm_name_);
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

        // Default fallback
        template<uint64_t N>
        inline void apply_gate_single_task(
                const uint64_t task_id,
                const Gate::UnitaryGateN <N, precision_t> &gate
        ) {
            const auto &qubits = gate.affect_args_;
            uarray_t<N> qubits_sorted;
            std::copy(gate.affect_args_.begin(), gate.affect_args_.end(), qubits_sorted);
            auto ind = index(task_id, qubit_num_, qubits, qubits_sorted);
            constexpr uint64_t vec_slice_sz = 1ULL << N;
            marray_t<precision_t, vec_slice_sz> tmp_arr;
#pragma unroll
            for (size_t i = 0; i < vec_slice_sz; ++i) {
                tmp_arr[i] = state_vector_[ind[i]];
                state_vector_[ind[i]] = static_cast<mat_entry_t<precision_t>>(0);
            }
#pragma unroll
            for (size_t i = 0; i < vec_slice_sz; ++i) {
#pragma unroll
                for (size_t j = 0; j < vec_slice_sz; ++j) {
                    state_vector_[ind[i]] += tmp_arr[j] * gate.mat_[j];
                }
            }
        }


        inline void apply_gate_single_task(
                const uint64_t task_id,
                const Gate::UnitaryGateN<1, precision_t> &gate
        ) {
            auto ind = index(task_id, qubit_num_, gate.targ_);
            auto tmp_arr_1 = marray_t<precision_t, 2>();
            tmp_arr_1[0] = state_vector_[ind[0]];
            tmp_arr_1[1] = state_vector_[ind[1]];
            state_vector_[ind[0]] = gate.mat_[0 * 2 + 0] * tmp_arr_1[0] + gate.mat_[0 * 2 + 1] * tmp_arr_1[1];
            state_vector_[ind[1]] = gate.mat_[1 * 2 + 0] * tmp_arr_1[0] + gate.mat_[1 * 2 + 1] * tmp_arr_1[1];
        }

    protected:
        std::complex<precision_t> *state_vector_ = nullptr;
        uint64_t qubit_num_ = 0;
        std::vector<GateBridgeEntry < precision_t>> gate_vec_;
    };
}

#endif //SIMULATION_BACKENDS_SIMULATOR_H

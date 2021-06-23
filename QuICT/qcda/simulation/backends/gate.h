//
// Created by Ci Lei on 2021-06-23.
//

#ifndef SIMULATION_BACKENDS_GATE_H
#define SIMULATION_BACKENDS_GATE_H

#include <complex>
#include <array>
#include <cstdint>
#include "utility.h"

namespace QuICT {
    namespace Gate {
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        // Basic Gate Declaration
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        // Single qubit non-parameterized gate
        struct SimpleGate {
            uint64_t targ_;

            SimpleGate(uint64_t targ) : targ_(targ) {}
        };

        /*
         * Multiple qubit non-parameterized gate.
         * Non-parameterized gate's matrix is fixed.
         * So we do not need this->mat_, but can directly use it.
         * For example, X gate just execute std::swap, we never need
         * a matrix for such gates. So it is up to the subclasses to
         * determine whether it need a mat_ member.
         * */
        template<uint64_t N>
        struct SimpleGateN {
            uarray_t <N> affect_args_;

            template<typename _rand_iter>
            SimpleGateN(_rand_iter qubit_begin, _rand_iter qubit_end) {
                std::copy(qubit_begin, qubit_end, affect_args_.begin());
            }
        };

        // Single qubit unitary gate
        template<typename precision_t>
        struct UnitaryGate {
            uint64_t targ_;
            marray_t<precision_t, 4> mat_;

            // this->mat_ is initialized by subclasses

            UnitaryGate(uint64_t targ) : targ_(targ) {}
        };

        // Multiple qubit unitary gate
        template<uint64_t N, typename precision_t>
        struct UnitaryGateN {
            uarray_t <N> affect_args_;
            marray_t<precision_t, 1ULL << (N << 1ULL)> mat_;

            // this->mat_ is initialized by subclasses
            template<typename _rand_iter>
            UnitaryGateN(_rand_iter qubit_begin, _rand_iter qubit_end) {
                std::copy(qubit_begin, qubit_end, affect_args_.begin());
            }
        };


        // Single qubit diagonal gate
        // Diagonal gate
        template<typename precision_t>
        struct DiagonalGate {
            uint64_t targ_;
            marray_t<precision_t, 2> diagonal_;

            // this->diagonal_ is initialized by subclasses

            DiagonalGate(uint64_t targ) : targ_(targ) {}
        };


        // Multiple qubit diagonal gate
        template<uint64_t N, typename precision_t>
        struct DiagonalGateN {
            uarray_t <N> affect_args_;
            marray_t<precision_t, 1ULL << N> diagonal_;

            // this->diagonal_ is initialized by subclasses
            template<typename _rand_iter>
            DiagonalGateN(_rand_iter qubit_begin, _rand_iter qubit_end) {
                std::copy(qubit_begin, qubit_end, affect_args_.begin());
            }
        };

        // Controlled gate of 2 qubits(i.e. 1 control bit and 1 target bit).
        template<typename precision_t>
        struct ControlledUnitaryGate : UnitaryGate<precision_t> {
            uint64_t carg_;

            // this->mat_ is initialized by subclasses
            ControlledUnitaryGate(uint64_t carg, uint64_t targ)
                    : carg_(carg), UnitaryGate<precision_t>(targ) {}
        };

        // Multiple bits controlled unitary gates are not currently used.
        // So we do not write a class for them.

        // Controlled diagonal gate of 2 qubits(i.e. 1 control bit and 1 target bit).
        template<typename precision_t>
        struct ControlledDiagonalGate : DiagonalGate<precision_t> {
            uint64_t carg_;

            // this->diagonal_ is initialized by subclasses
            ControlledDiagonalGate(uint64_t carg, uint64_t targ)
                    : carg_(carg), DiagonalGate<precision_t>(targ) {}
        };


        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        // Specific Gate Declaration
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        template<typename precision_t>
        struct CrzGate : ControlledDiagonalGate<precision_t> {
            CrzGate(uint64_t carg, uint64_t targ, precision_t parg)
                    : ControlledDiagonalGate<precision_t>(carg, targ) {
                this->diagonal_[0] = std::exp(static_cast<mat_entry_t<precision_t>>(-1j * parg / 2));
                this->diagonal_[1] = std::exp(static_cast<mat_entry_t<precision_t>>(1j * parg / 2));
            }
        };

        template<typename precision_t>
        struct HGate : SimpleGate {
            mat_entry_t <precision_t> sqrt2_inv = static_cast<std::complex<precision_t>>(1.0 / sqrt(2));

            HGate(uint64_t targ) : SimpleGate(targ) {}
        };

        template<typename precision_t>
        struct ZGate : DiagonalGate<precision_t> {

            ZGate(uint64_t targ) : DiagonalGate<precision_t>(targ) {
                this->diagonal_[0] = static_cast<mat_entry_t<precision_t>>(1);
                this->diagonal_[1] = static_cast<mat_entry_t<precision_t>>(-1);
            }
        };


        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        // Type Helpers
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        template<typename gate_t>
        struct is_diagonal_gate {
            constexpr static bool value = is_base_of_template<DiagonalGate, gate_t>::value &&
                                          !is_base_of_template<ControlledDiagonalGate, gate_t>::value;
        };

        template<typename gate_t>
        struct is_simple_gate {
            constexpr static bool value = std::is_base_of<SimpleGate, gate_t>::value;
        };

        template<typename gate_t>
        struct is_single_unitary_gate {
            constexpr static bool value = is_base_of_template<UnitaryGate, gate_t>::value &&
                                          !is_base_of_template<ControlledUnitaryGate, gate_t>::value;
        };

        template<typename gate_t>
        struct is_single_bit {
            constexpr static bool value = is_simple_gate<gate_t>::value ||
                                          is_diagonal_gate<gate_t>::value ||
                                          is_single_unitary_gate<gate_t>::value;
        };

        template<typename gate_t>
        struct is_controlled_unitary_gate {
            constexpr static bool value = is_base_of_template<ControlledUnitaryGate, gate_t>::value;
        };

        template<typename gate_t>
        struct is_controlled_diagonal_gate {
            constexpr static bool value = is_base_of_template<ControlledDiagonalGate, gate_t>::value;
        };

        template<typename gate_t>
        struct is_controlled_2_bit {
            constexpr static bool value = is_controlled_unitary_gate<gate_t>::value ||
                                          is_controlled_diagonal_gate<gate_t>::value;
        };

        template<typename gate_t>
        struct gate_qubit_num {
            constexpr static uint64_t value = []() -> auto {
                if constexpr(is_single_bit<gate_t>::value) {
                    return 1;
                } else if constexpr(is_controlled_2_bit<gate_t>::value) {
                    return 2;
                } else {
                    return std::declval<gate_t>().affect_args_.size();
                }
            }();
        };
    }
}

#endif //SIMULATION_BACKENDS_GATE_H

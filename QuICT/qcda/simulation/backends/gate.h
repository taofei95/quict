//
// Created by Ci Lei on 2021-06-23.
//

#ifndef SIMULATION_BACKENDS_GATE_H
#define SIMULATION_BACKENDS_GATE_H

#include <complex>
#include <array>
#include <cstdint>
#include <string>
#include "utility.h"


namespace QuICT {
    namespace Details {
        // Black magic from
        // https://stackoverflow.com/questions/1005476/how-to-detect-whether-there-is-a-specific-member-variable-in-class

#define DEFINE_MEMBER_CHECKER(member) \
            template<typename T, typename = int> \
            struct has_member_ ## member : std::false_type {}; \
            template<typename T>      \
            struct has_member_ ## member <T, decltype((void)T:: member, 0)> : std::true_type{};

        DEFINE_MEMBER_CHECKER(affect_args_);
        DEFINE_MEMBER_CHECKER(mat_);
        DEFINE_MEMBER_CHECKER(diagonal_);
        DEFINE_MEMBER_CHECKER(carg_);

#undef DEFINE_MEMBER_CHECKER
    }

    namespace Gate {
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        // Basic Gate Declaration
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

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

        // Single qubit non-parameterized gate
        template<>
        struct SimpleGateN<1> {
            uint64_t targ_;

            SimpleGateN(uint64_t targ) : targ_(targ) {}
        };

        // Multiple qubit unitary gate
        template<uint64_t N, typename precision_t>
        struct UnitaryGateN {
            uarray_t <N> affect_args_;
            mat_entry_t <precision_t> *mat_;

            template<typename _qubit_iter>
            UnitaryGateN(_qubit_iter qubit_begin, _qubit_iter qubit_end) {
                std::copy(qubit_begin, qubit_end, affect_args_.begin());
            }

            template<typename _qubit_iter, typename _mat_iter>
            UnitaryGateN(_qubit_iter qubit_begin, _mat_iter mat_begin) {
                std::copy(qubit_begin, qubit_begin + N, affect_args_.begin());
                mat_ = new mat_entry_t<precision_t>[1ULL << (N << 1)];
                std::copy(mat_begin, mat_begin + (1ULL << (N << 1)), mat_);
            }

            ~UnitaryGateN() {
                delete[] mat_;
            }
        };

        // Single qubit unitary gate
        template<typename precision_t>
        struct UnitaryGateN<1, precision_t> {
            uint64_t targ_;
            mat_entry_t <precision_t> *mat_;

            UnitaryGateN(uint64_t targ) : targ_(targ) {}

            template<typename _mat_iter>
            UnitaryGateN(uint64_t targ, _mat_iter mat_begin) : targ_(targ) {
                mat_ = new mat_entry_t<precision_t>[4];
                std::copy(mat_begin, mat_begin + 4, mat_);
            }

            ~UnitaryGateN() {
                delete[] mat_;
            }
        };


        // Multiple qubit diagonal gate
        template<uint64_t N, typename precision_t>
        struct DiagonalGateN {
            uarray_t <N> affect_args_;
            mat_entry_t <precision_t> *diagonal_;

            // this->diagonal_ is initialized by subclasses
            template<typename _qubit_iter>
            DiagonalGateN(_qubit_iter qubit_begin) {
                std::copy(qubit_begin, qubit_begin + N, affect_args_.begin());
            }
        };

        // Single qubit diagonal gate
        // Diagonal gate
        template<typename precision_t>
        struct DiagonalGateN<1, precision_t> {
            uint64_t targ_;
            mat_entry_t <precision_t> *diagonal_;

            // this->diagonal_ is initialized by subclasses

            DiagonalGateN(uint64_t targ) : targ_(targ) {}
        };

        // Controlled gate of 2 qubits(i.e. 1 control bit and 1 target bit).
        template<typename precision_t>
        struct ControlledUnitaryGate : UnitaryGateN<1, precision_t> {
            uint64_t carg_;

            // this->mat_ is initialized by subclasses
            ControlledUnitaryGate(uint64_t carg, uint64_t targ)
                    : carg_(carg), UnitaryGateN<1, precision_t>(targ) {}
        };

        // Multiple bits controlled unitary gates are not currently used.
        // So we do not write a class for them.

        // Controlled diagonal gate of 2 qubits(i.e. 1 control bit and 1 target bit).
        template<typename precision_t>
        struct ControlledDiagonalGate : DiagonalGateN<1, precision_t> {
            uint64_t carg_;

            // this->diagonal_ is initialized by subclasses
            ControlledDiagonalGate(uint64_t carg, uint64_t targ)
                    : carg_(carg), DiagonalGateN<1, precision_t>(targ) {}
        };


        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        // Specific Gate Declaration
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        template<typename precision_t>
        struct CrzGate : ControlledDiagonalGate<precision_t> {
            CrzGate(uint64_t carg, uint64_t targ, precision_t parg)
                    : ControlledDiagonalGate<precision_t>(carg, targ) {
                this->diagonal_ = new mat_entry_t<precision_t>[2];
                this->diagonal_[0] = std::exp(static_cast<mat_entry_t<precision_t>>(-1j * parg / 2));
                this->diagonal_[1] = std::exp(static_cast<mat_entry_t<precision_t>>(1j * parg / 2));
            }

            ~CrzGate() {
                delete[] this->diagonal_;
            }
        };

        template<typename precision_t>
        struct HGate : SimpleGateN<1> {
//            std::complex<precision_t> sqrt2_inv = static_cast<std::complex<precision_t>>(1.0 / sqrt(2));
            static constexpr auto sqrt2_inv =
                    static_cast<mat_entry_t <precision_t>>(
                            1.0 / 1.4142135623730950488016887242096980785696718753769480731766797379);

            HGate(uint64_t targ) : SimpleGateN(targ) {}
        };

        template<typename precision_t>
        struct ZGate : DiagonalGateN<1, precision_t> {

            ZGate(uint64_t targ) : DiagonalGateN<1, precision_t>(targ) {
                this->diagonal_ = new mat_entry_t<precision_t>[2];
                this->diagonal_[0] = static_cast<mat_entry_t<precision_t>>(1);
                this->diagonal_[1] = static_cast<mat_entry_t<precision_t>>(-1);
            }

            ~ZGate() {
                delete[] this->diagonal_;
            }
        };


        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        // Type Helpers
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        template<class gate_t>
        struct gate_qubit_num {
            constexpr static uint64_t value = []() {
                if constexpr(Details::has_member_affect_args_<gate_t>::value) {
                    return gate_t::affect_args_.size();
                } else if constexpr(Details::has_member_carg_<gate_t>::value) {
                    return 2;
                } else {
                    return 1;
                }
            }();
        };

        template<class gate_t>
        struct gate_is_diagonal {
            constexpr static bool value = Details::has_member_diagonal_<gate_t>::value;
        };

        template<class gate_t>
        struct gate_has_mat_repr {
            constexpr static bool value = Details::has_member_mat_<gate_t>::value;
        };

    }
}

#endif //SIMULATION_BACKENDS_GATE_H

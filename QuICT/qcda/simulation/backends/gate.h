//
// Created by Ci Lei on 2021-06-23.
//

#ifndef SIMULATION_BACKENDS_GATE_H
#define SIMULATION_BACKENDS_GATE_H

#include <complex>
#include <array>
#include <cstdint>
#include <string>
#include <vector>
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

        explicit SimpleGateN(uint64_t targ) : targ_(targ) {}
    };

    // Multiple qubit unitary gate
    template<uint64_t N, typename Precision>
    struct UnitaryGateN {
        uarray_t <N> affect_args_;
        Precision *mat_real_ = nullptr;
        Precision *mat_imag_ = nullptr;

        explicit UnitaryGateN(const std::vector<uint64_t> &args,
                              const std::shared_ptr<std::complex<Precision>[]> data_ptr) {
            std::copy(args.begin(), args.end(), this->affect_args_.begin());

            this->mat_real_ = new Precision[1 << (N << 1)];
            this->mat_imag_ = new Precision[1 << (N << 1)];
            for (int i = 0; i < 1 << (N << 1); i++) {
                this->mat_real_[i] = data_ptr[i].real();
                this->mat_imag_[i] = data_ptr[i].imag();
            }
        }

        ~UnitaryGateN() {
            delete_all_and_set_null(this->mat_real_);
            delete_all_and_set_null(this->mat_imag_);
        }
    };

    // Single qubit unitary gate
    template<typename Precision>
    struct UnitaryGateN<1, Precision> {
        uint64_t targ_;
        Precision *mat_real_ = nullptr;
        Precision *mat_imag_ = nullptr;

        explicit UnitaryGateN(uint64_t targ) : targ_(targ) {}

        explicit UnitaryGateN(uint64_t targ, const std::shared_ptr<std::complex<Precision>[]> data_ptr) : targ_(targ) {
            this->mat_real_ = new Precision[4];
            this->mat_imag_ = new Precision[4];
            for (int i = 0; i < 4; i++) {
                mat_real_[i] = data_ptr[i].real();
                mat_imag_[i] = data_ptr[i].imag();
            }
        }

        ~UnitaryGateN() {
            delete_all_and_set_null(this->mat_real_);
            delete_all_and_set_null(this->mat_imag_);
        }
    };


    // Multiple qubit diagonal gate
    template<uint64_t N, typename Precision>
    struct DiagonalGateN {
        uarray_t <N> affect_args_;
        Precision *diagonal_real_ = nullptr;
        Precision *diagonal_imag_ = nullptr;

        explicit DiagonalGateN(const uarray_t <N> &affect_args,
                               const std::shared_ptr<std::complex<Precision>[]> data_ptr) {
            std::copy(affect_args.begin(), affect_args.end(), this->affect_args_.begin());
            this->diagonal_real_ = new Precision[1ULL << N];
            this->diagonal_imag_ = new Precision[1ULL << N];
            for (uint64_t i = 0; i < (1ULL << N); ++i) {
                this->diagonal_real_[i] = data_ptr[i].real();
                this->diagonal_imag_[i] = data_ptr[i].imag();
            }
        }

        ~DiagonalGateN() {
            delete_all_and_set_null(this->diagonal_real_);
            delete_all_and_set_null(this->diagonal_imag_);
        }
    };

    // Single qubit diagonal gate
    // Diagonal gate
    template<typename Precision>
    struct DiagonalGateN<1, Precision> {
        uint64_t targ_;
        Precision *diagonal_real_ = nullptr;
        Precision *diagonal_imag_ = nullptr;

        // this->diagonal_ is initialized by subclasses

        explicit DiagonalGateN(uint64_t targ) : targ_(targ) {}

        explicit DiagonalGateN(uint64_t targ, const std::shared_ptr<std::complex<Precision>[]> data_ptr) : targ_(targ) {
            this->diagonal_real_ = new Precision[2];
            this->diagonal_imag_ = new Precision[2];
            for (int i = 0; i < 2; ++i) {
                this->diagonal_real_[i] = data_ptr[i].real();
                this->diagonal_imag_[i] = data_ptr[i].imag();
            }
        }

        ~DiagonalGateN() {
            delete_all_and_set_null(this->diagonal_real_);
            delete_all_and_set_null(this->diagonal_imag_);
        }
    };

    // Controlled gate of 2 qubits(i.e. 1 control bit and 1 target bit).
    template<typename Precision>
    struct ControlledUnitaryGate : UnitaryGateN<1, Precision> {
        uint64_t carg_;

        explicit ControlledUnitaryGate(uint64_t carg, uint64_t targ)
                : carg_(carg), UnitaryGateN<1, Precision>(targ) {}

        explicit ControlledUnitaryGate(uint64_t carg, uint64_t targ,
                                       const std::shared_ptr<std::complex<Precision>[]> data_ptr)
                : carg_(carg), UnitaryGateN<1, Precision>(targ, data_ptr) {}
    };

    // Multiple bits controlled unitary gates are not currently used.
    // So we do not write a class for them.

    // Controlled diagonal gate of 2 qubits(i.e. 1 control bit and 1 target bit).
    template<typename Precision>
    struct ControlledDiagonalGate : DiagonalGateN<1, Precision> {
        uint64_t carg_;

        explicit ControlledDiagonalGate(uint64_t carg, uint64_t targ)
                : carg_(carg), DiagonalGateN<1, Precision>(targ) {}

        explicit ControlledDiagonalGate(uint64_t carg, uint64_t targ,
                                        const std::shared_ptr<std::complex<Precision>[]> data_ptr)
                : carg_(carg), DiagonalGateN<1, Precision>(targ, data_ptr) {}
    };


    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // Specific Gate Declaration
    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

    template<typename Precision>
    struct HGate : SimpleGateN<1> {
        static constexpr std::complex<Precision> sqrt2_inv =
                static_cast<std::complex<Precision>>(
                        1.0 / 1.414213562373095048801688724209698);

        explicit HGate(uint64_t targ) : SimpleGateN(targ) {}
    };

    template<typename Precision>
    struct XGate : SimpleGateN<1> {
        explicit XGate(uint64_t targ) : SimpleGateN<1>(targ) {}
    };


    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // Type Helpers
    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

    template<typename Gate>
    struct gate_qubit_num {
        constexpr static uint64_t value = []() {
            if constexpr(Details::has_member_affect_args_<Gate>::value) {
                // Compile Error: Invalid use of non-static member affect_args_ (2021-09-25)
                return Gate::affect_args_.size();
            } else if constexpr(Details::has_member_carg_<Gate>::value) {
                return 2;
            } else {
                return 1;
            }
        }();
    };

    template<typename Gate>
    inline constexpr uint64_t gate_qubit_num_v = gate_qubit_num<Gate>::value;

//    template<class Gate>
//    struct gate_is_diagonal {
//        constexpr static bool value = Details::has_member_diagonal_<Gate>::value;
//    };
//
//    template<class Gate>
//    struct gate_has_mat_repr {
//        constexpr static bool value = Details::has_member_mat_<Gate>::value;
//    };

    template<typename Gate>
    struct is_simple_gate {
        static constexpr bool value =
                std::is_base_of_v<SimpleGateN<1>, Gate> ||
                std::is_base_of_v<SimpleGateN<2>, Gate>;

    };

    template<typename Gate>
    inline constexpr bool is_simple_gate_v = is_simple_gate<Gate>::value;

    template<typename Gate>
    struct is_diag_n_gate {
        static constexpr bool value =
                std::is_base_of_v<DiagonalGateN<1, double>, Gate> ||
                std::is_base_of_v<DiagonalGateN<2, double>, Gate> ||
                std::is_base_of_v<DiagonalGateN<1, float>, Gate> ||
                std::is_base_of_v<DiagonalGateN<2, float>, Gate>;
    };

    template<typename Gate>
    inline constexpr bool is_diag_n_gate_v = is_diag_n_gate<Gate>::value;

    template<typename Gate>
    struct is_ctrl_diag_gate {
        static constexpr bool value =
                std::is_base_of_v<ControlledDiagonalGate<double>, Gate> ||
                std::is_base_of_v<ControlledDiagonalGate<float>, Gate>;
    };

    template<typename Gate>
    inline constexpr bool is_ctrl_diag_gate_v = is_ctrl_diag_gate<Gate>::value;

    template<typename Gate>
    struct is_unitary_n_gate {
        static constexpr bool value =
                std::is_same_v<Gate, UnitaryGateN<1, double>> ||
                std::is_same_v<Gate, UnitaryGateN<2, double>> ||
                std::is_same_v<Gate, UnitaryGateN<1, float>> ||
                std::is_same_v<Gate, UnitaryGateN<2, float>>;
    };

    template<typename Gate>
    inline constexpr bool is_unitary_n_gate_v = is_unitary_n_gate<Gate>::value;

    template<typename Gate>
    struct is_ctrl_unitary_gate {
        static constexpr bool value =
                std::is_same_v<Gate, ControlledUnitaryGate<double>> ||
                std::is_same_v<Gate, ControlledUnitaryGate<float>>;
    };

    template<typename Gate>
    inline constexpr bool is_ctrl_unitary_gate_v = is_ctrl_unitary_gate<Gate>::value;
}

#endif //SIMULATION_BACKENDS_GATE_H

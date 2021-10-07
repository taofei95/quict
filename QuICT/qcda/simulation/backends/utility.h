//
// Created by Ci Lei on 2021-06-23.
//

#ifndef SIMULATION_BACKENDS_UTILITY_H
#define SIMULATION_BACKENDS_UTILITY_H

#include <cstdint>
#include <array>
#include <complex>
#include <type_traits>
#include <chrono>
#include <memory>
#include <vector>
#include <cassert>
#include <fstream>
#include <immintrin.h>

// v1 * v2 == (v1r * v2r - v1i * v2i) + (v1i * v2r + v1r * v2i)*J
#define COMPLEX_YMM_MUL(v1r, v1i, v2r, v2i, res_r, res_i) \
    res_r = _mm256_mul_pd(v1r, v2r);\
    res_r = _mm256_fnmadd_pd(v1i, v2i, res_r);\
    res_i = _mm256_mul_pd(v1i, v2r);\
    res_i = _mm256_fmadd_pd(v1r, v2i, res_i);


#define STRIDE_2_LOAD_ODD_PD(from_addr, to_reg, tmp1, tmp2) \
    tmp1 = _mm256_loadu_pd(from_addr);\
    tmp2 = _mm256_loadu_pd(((double*)(from_addr)) + 4);\
    to_reg = _mm256_shuffle_pd(tmp1, tmp2, 0b1111);\
    to_reg = _mm256_permute4x64_pd(to_reg, 0b11011000);


#define STRIDE_2_STORE_ODD_PD(to_addr, from_reg, tmp_arr) \
    _mm256_storeu_pd(tmp_arr, from_reg);\
    ((double*)(to_addr))[1] = ((double*)(tmp_arr))[0];\
    ((double*)(to_addr))[3] = ((double*)(tmp_arr))[1];\
    ((double*)(to_addr))[5] = ((double*)(tmp_arr))[2];\
    ((double*)(to_addr))[7] = ((double*)(tmp_arr))[3];


namespace QuICT {
    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // Time Measurement
    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

    template<typename Callable, typename ...Arg, typename TimeUnit=std::chrono::microseconds>
    uint64_t time_elapse(Callable callable, Arg &&... args) {
        using namespace std;
        auto start_time = chrono::steady_clock::now();
        callable(args...);
        auto end_time = chrono::steady_clock::now();
        auto cnt = chrono::duration_cast<TimeUnit>(end_time - start_time).count();
        return cnt;
    }

    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // Some Definitions
    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

    enum gate_category {
        diag_1,
        diag_2,
        ctrl_diag,
        unitary_1,
        unitary_2,
        ctrl_unitary,
        special_h,
        special_x
    };

    const std::map<std::string, gate_category> dispatcher = {
            {"special_h",    gate_category::special_h},
            {"special_x",    gate_category::special_x},
            {"diag_1",       gate_category::diag_1},
            {"diag_2",       gate_category::diag_2},
            {"ctrl_diag",    gate_category::ctrl_diag},
            {"unitary_1",    gate_category::unitary_1},
            {"unitary_2",    gate_category::unitary_2},
            {"ctrl_unitary", gate_category::ctrl_unitary}
    };

    template<typename Precision>
    inline std::pair<Precision *, Precision *> separate_complex(
            uint64_t q_state_bit_num,
            const std::complex<Precision> *c_arr
    ) {
        auto len = 1ULL << q_state_bit_num;
        auto ptr = new Precision[len << 1ULL];
        auto real = ptr;
        auto imag = &ptr[len];
        for (uint64_t i = 0; i < len; i += 4) {
            real[i] = c_arr[i].real();
            imag[i] = c_arr[i].imag();

            real[i + 1] = c_arr[i + 1].real();
            imag[i + 1] = c_arr[i + 1].imag();

            real[i + 2] = c_arr[i + 2].real();
            imag[i + 2] = c_arr[i + 2].imag();

            real[i + 3] = c_arr[i + 3].real();
            imag[i + 3] = c_arr[i + 3].imag();
        }
        return {real, imag};
    }

    template<typename Precision>
    inline void combine_complex(
            uint64_t q_state_bit_num,
            const Precision *real,
            const Precision *imag,
            std::complex<Precision> *res
    ) {
        auto len = 1ULL << q_state_bit_num;
        for (uint64_t i = 0; i < len; i += 4) {
            res[i] = {real[i], imag[i]};
            res[i + 1] = {real[i + 1], imag[i + 1]};
            res[i + 2] = {real[i + 2], imag[i + 2]};
            res[i + 3] = {real[i + 3], imag[i + 3]};
        }
    }


    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // Template Class Derive Check Helpers
    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

//    //https://stackoverflow.com/questions/34672441/stdis-base-of-for-template-classes
//    template<template<typename...> class _Base, typename..._Derive_Args>
//    std::true_type is_base_of_template_impl(const _Base<_Derive_Args...> *);
//
//    template<template<typename...> class _Base>
//    std::false_type is_base_of_template_impl(...);
//
//    template<template<typename...> class _Base, typename _Derived>
//    using is_base_of_template = decltype(is_base_of_template_impl<_Base>(std::declval<_Derived *>()));


    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // Pointer helpers
    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

    template<typename T>
    inline void delete_and_set_null(T &ptr) {
        if (ptr) {
            delete ptr;
            ptr = nullptr;
        }
    }

    template<typename T>
    inline void delete_all_and_set_null(T &ptr) {
        if (ptr) {
            delete[] ptr;
            ptr = nullptr;
        }
    }

    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // Data type aliases
    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

    template<uint64_t N>
    using uarray_t = std::array<uint64_t, N>;

    template<typename Precision>
    using mat_entry_t = std::complex<Precision>;

    template<typename Precision, uint64_t N>
    using marray_t = std::array<std::complex<Precision>, N>;


    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // Helper Class for Receiving Data from Python
    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

    template<typename Precision>
    class GateDescription {
    public:
        std::string gate_name_;
        std::vector<uint64_t> affect_args_;
        std::vector<std::complex<Precision>> data_ptr_;

        template<typename _u64_vec_T>
        GateDescription(
                const char *gate_name,
                _u64_vec_T &&affect_args
        ) :
                gate_name_(gate_name),
                affect_args_(std::move(affect_args)) {}

        template<typename _u64_vec_T>
        GateDescription(
                const char *gate_name,
                _u64_vec_T &&affect_args,
                std::vector<std::complex<Precision>> &&data_ptr
        ) :
                gate_name_(gate_name),
                affect_args_(std::move(affect_args)),
                data_ptr_(std::move(data_ptr)) {}
    };

    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // Helper functions to create indices array
    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


    template<
            uint64_t N,
            typename std::enable_if<(N > 1), int>::type dummy = 0
    >
    inline uint64_t index0(
            const uint64_t task_id,
            const uint64_t qubit_num,
            const uarray_t<N> &qubits,
            const uarray_t<N> &qubits_sorted
    ) {
        uint64_t ret = task_id;
        for (int64_t i = N - 1; i >= 0; --i) {
            uint64_t pos = qubit_num - 1 - qubits_sorted[i];
            uint64_t tail = ret & ((1ULL << pos) - 1);
            ret = ret >> pos << (pos + 1) | tail;
        }
        return ret;
    }

    template<
            uint64_t N = 1,
            typename std::enable_if<(N == 1), int>::type dummy = 0
    >
    inline uint64_t index0(
            const uint64_t task_id,
            const uint64_t qubit_num,
            const uint64_t targ
    ) {
        uint64_t pos = qubit_num - 1 - targ;
        uint64_t tail = task_id & ((1ULL << pos) - 1);
        uint64_t ret = task_id >> pos << (pos + 1) | tail;
        return ret;
    }

    template<
            uint64_t N,
            typename std::enable_if<(N > 2), int>::type dummy = 0
    >
    inline uarray_t<1ULL << N> index(
            const uint64_t task_id,
            const uint64_t qubit_num,
            const uarray_t<N> &qubits,
            const uarray_t<N> &qubits_sorted
    ) {
        auto ret = uarray_t<1ULL << N>();
        ret[0] = index0(task_id, qubit_num, qubits, qubits_sorted);

        ret[1] = ret[0] | (1ULL << (qubit_num - 1 - qubits[N - 1]));
        ret[2] = ret[0] | (1ULL << (qubit_num - 1 - qubits[N - 2]));
        ret[3] = ret[1] | (1ULL << (qubit_num - 1 - qubits[N - 2]));

        for (int64_t i = 2; i < N; ++i) {
            const auto half_cnt = 1ULL << i;
            const auto tail = 1ULL << (qubit_num - 1 - qubits[N - 1 - i]);
            __m256d tail4 = _mm256_set1_pd(*((double *) &tail));

            for (uint64_t j = 0; j < half_cnt; j += 4) {
                __m256d cur = _mm256_loadu_pd((double *) &ret[j]);
                __m256d res = _mm256_or_pd(cur, tail4);
                _mm256_storeu_pd((double *) &ret[half_cnt + j], res);
            }
        }
        return ret;
    }

    template<
            uint64_t N = 2,
            typename std::enable_if<(N == 2), int>::type dummy = 0
    >
    inline uarray_t<4> index(
            const uint64_t task_id,
            const uint64_t qubit_num,
            const uarray_t<2> &qubits,
            const uarray_t<2> &qubits_sorted) {
        auto ret = uarray_t<4>();

        const uint64_t pos0 = qubit_num - 1 - qubits_sorted[1];
        const uint64_t pos1 = qubit_num - 2 - qubits_sorted[0];
        const uint64_t msk0 = (1ULL << pos0) - 1;
        const uint64_t msk1 = (1ULL << pos1) - 1;
        // 0 ... pos0 ... pos1 ... q-1
        // [    ][       ][          ] ->
        // [    ]0[        ]0[         ]

        const uint64_t part0 = task_id & msk0;
        const uint64_t part1 = task_id & (msk1 ^ msk0);
        const uint64_t part2 = task_id & (~msk1);
        const uint64_t init = part0 | (part1 << 1) | (part2 << 2);

        ret[0] = init;
        ret[1] = ret[0] | (1ULL << (qubit_num - 1 - qubits[N - 1]));
        ret[2] = ret[0] | (1ULL << (qubit_num - 1 - qubits[N - 2]));
        ret[3] = ret[1] | (1ULL << (qubit_num - 1 - qubits[N - 2]));

        return ret;
    }

    template<
            uint64_t N = 1,
            typename std::enable_if<(N == 1), int>::type dummy = 0
    >
    inline uarray_t<2> index(
            const uint64_t task_id,
            const uint64_t qubit_num,
            const uint64_t targ
    ) {
        uarray_t<2> ret = uarray_t<2>();
        uint64_t pos = qubit_num - 1 - targ;
        uint64_t tail = task_id & ((1ULL << pos) - 1);
        ret[0] = task_id >> pos << (pos + 1) | tail;
        ret[1] = ret[0] | (1ULL << pos);
        return ret;
    }

    namespace Test {
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        // Test helper
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        template<
                template<typename...> class _sim_T,
                typename _str_T,
                typename Precision
        >
        void test_by_data_file(
                _str_T data_name,
                _sim_T<Precision> &simulator,
                double eps = 1e-6
        ) {
            using namespace std;

            cout << simulator.name() << " " << "Testing by " << data_name << endl;

            fstream fs;
            fs.open(data_name, ios::in);
            if (!fs) {
                ASSERT_EQ(0, 1) << "Open " << data_name << " description failed.";
            }
            uint64_t qubit_num;
            fs >> qubit_num;
            string gate_name;
            std::vector<QuICT::GateDescription<Precision>> gate_desc_vec;
            while (fs >> gate_name) {
                if (gate_name == "__TERM__") {
                    break;
                }
                uint64_t carg;
                uint64_t targ;

                if (gate_name == "special_h") {
                    fs >> targ;
                    gate_desc_vec.emplace_back(
                            "special_h",
                            std::vector<uint64_t>{targ}
                    );
                } else if (gate_name == "special_x") {
                    fs >> targ;
                    gate_desc_vec.emplace_back(
                            "special_x",
                            std::vector<uint64_t>{targ}
                    );
                } else if (gate_name == "unitary_1") {
                    fs >> targ;
                    auto mat = std::vector<std::complex<Precision>>(4);
                    for (int i = 0; i < 4; i++) {
                        double re, im;
                        char sign, img_label;
                        fs >> re >> sign >> im >> img_label;
                        mat[i] = std::complex<double>(re, sign == '+' ? im : -im);
                    }

                    gate_desc_vec.emplace_back(
                            "unitary_1",
                            std::vector<uint64_t>{targ},
                            std::move(mat)
                    );
                } else if (gate_name == "unitary_2") {
                    fs >> carg >> targ;
                    auto mat = std::vector<std::complex<Precision>>(16);
                    for (int i = 0; i < 16; i++) {
                        double re, im;
                        char sign, img_label;
                        fs >> re >> sign >> im >> img_label;
                        mat[i] = std::complex<double>(re, sign == '+' ? im : -im);
                    }

                    gate_desc_vec.emplace_back(
                            "unitary_2",
                            std::vector<uint64_t>{carg, targ},
                            std::move(mat)
                    );
                } else if (gate_name == "ctrl_unitary") {
                    fs >> carg >> targ;
                    auto mat = std::vector<std::complex<Precision>>(4);
                    for (int i = 0; i < 4; i++) {
                        double re, im;
                        char sign, img_label;
                        fs >> re >> sign >> im >> img_label;
                        mat[i] = std::complex<double>(re, sign == '+' ? im : -im);
                    }
                    gate_desc_vec.emplace_back(
                            "ctrl_unitary",
                            std::vector<uint64_t>{carg, targ},
                            std::move(mat)
                    );
                } else if (gate_name == "diag_1") {
                    fs >> targ;
                    auto diag = std::vector<std::complex<Precision>>(2);
                    for (int i = 0; i < 2; ++i) {
                        double re, im;
                        char sign, img_label;
                        fs >> re >> sign >> im >> img_label;
                        diag[i] = std::complex<double>(re, sign == '+' ? im : -im);
                    }
                    gate_desc_vec.emplace_back(
                            "diag_1",
                            std::vector<uint64_t>{targ},
                            std::move(diag)
                    );
                } else if (gate_name == "ctrl_diag") {
                    fs >> carg >> targ;
                    auto diag = std::vector<std::complex<Precision>>(2);
                    for (int i = 0; i < 2; ++i) {
                        double re, im;
                        char sign, img_label;
                        fs >> re >> sign >> im >> img_label;
                        diag[i] = std::complex<double>(re, sign == '+' ? im : -im);
                    }
                    gate_desc_vec.emplace_back(
                            "ctrl_diag",
                            std::vector<uint64_t>{carg, targ},
                            std::move(diag)
                    );
                }
            }

            auto expect_state = new std::complex<double>[1ULL << qubit_num];

            double re, im;
            char sign, img_label;
            for (uint64_t i = 0; i < (1ULL << qubit_num); ++i) {
                fs >> re >> sign >> im >> img_label;
                expect_state[i] = std::complex<double>(re, sign == '+' ? im : -im);
            }

            std::complex<Precision> *state = simulator.run(qubit_num, gate_desc_vec);
            for (uint64_t i = 0; i < (1ULL << qubit_num); ++i) {
                ASSERT_NEAR(state[i].real(), expect_state[i].real(), eps) << "i = " << i;
                ASSERT_NEAR(state[i].imag(), expect_state[i].imag(), eps) << "i = " << i;
            }
            delete[] state;
            delete[] expect_state;
        }
    }
}

#endif //SIMULATION_BACKENDS_UTILITY_H

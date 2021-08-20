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
#include <cassert>
#include <immintrin.h>

#ifndef OMP_NPROC
#define OMP_NPROC 4
#endif

#define LLC_OPTIM
// #define LLC_OPTIM_2

namespace QuICT {
    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // Time Measurement
    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    template<typename Callable, typename ...Arg, typename TimeUnit=std::chrono::microseconds>
    uint64_t time_elapse(Callable callable, Arg ... args) {
        using namespace std;
        auto start_time = chrono::steady_clock::now();
        callable(args...);
        auto end_time = chrono::steady_clock::now();
        auto cnt = chrono::duration_cast<TimeUnit>(end_time - start_time).count();
        return cnt;
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
    // Data type aliases
    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

    template<uint64_t N>
    using uarray_t = std::array<uint64_t, N>;

    template<typename Precision>
    using mat_entry_t = std::complex<Precision>;

    template<typename Precision, uint64_t N>
    using marray_t = std::array<mat_entry_t<Precision>, N>;


    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // Helper Class for Receiving Data from Python
    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

    template<typename Precision>
    class GateDescription {
    public:
        // Not all members are used

        std::string qasm_name_;
        Precision parg_;
        std::vector<uint64_t> affect_args_;
        mat_entry_t<Precision> *data_ptr_;

        GateDescription(
                std::string qasm_name,
                std::vector<uint64_t> affect_args,
                Precision parg,
                mat_entry_t<Precision> *data_ptr
        ) : qasm_name_(qasm_name), affect_args_(affect_args), parg_(parg), data_ptr_(data_ptr) {}

        GateDescription(
                const char *qasm_name,
                std::vector<uint64_t> affect_args,
                Precision parg,
                mat_entry_t<Precision> *data_ptr
        ) : qasm_name_(qasm_name), affect_args_(affect_args), parg_(parg), data_ptr_(data_ptr) {}
    };

    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // Helper functions to create indices array
    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

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
        ret[0] = task_id;

        for (int64_t i = N - 1; i >= 0; --i) {
            uint64_t pos = qubit_num - 1 - qubits_sorted[i];
            uint64_t tail = ret[0] & ((1ULL << pos) - 1);
//            ret[0] = (ret[0] << 1) - tail;
            ret[0] = ret[0] >> pos << (pos + 1) | tail;
        }

        ret[1] = ret[0] | (1ULL << (qubit_num - 1 - qubits[N - 1]));
        ret[2] = ret[0] | (1ULL << (qubit_num - 1 - qubits[N - 2]));
        ret[3] = ret[1] | (1ULL << (qubit_num - 1 - qubits[N - 2]));

        for(int64_t i = 2; i < N; ++i)
        {
            const auto half_cnt = 1ULL << i;
            const auto tail = 1ULL << (qubit_num - 1 - qubits[N - 1 - i]);
            __m256d tail4 = _mm256_set1_pd(*((double *)&tail));

            for(uint64_t j = 0; j < half_cnt; j += 4)
            {
                __m256d cur = _mm256_loadu_pd((double *)&ret[j]);
                __m256d res = _mm256_or_pd(cur, tail4);
                _mm256_storeu_pd((double *)&ret[half_cnt+j], res);
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
            const uarray_t<2> &qubits_sorted)
    {
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
//        ret[0] = (task_id << 1) - tail;
        ret[1] = ret[0] | (1ULL << pos);
        return ret;
    }
}

#endif //SIMULATION_BACKENDS_UTILITY_H

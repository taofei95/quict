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

#ifndef OMP_NPROC
#define OMP_NPROC 4
#endif

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
    };

    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // Helper functions to create indices array
    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

    template<
            uint64_t N,
            typename std::enable_if<(N > 1), int>::type dummy = 0
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
            ret[0] = ret[0] >> pos << (pos + 1) | tail;
        }

        for (int64_t i = 0; i < N; ++i) {
            const auto half_cnt = 1ULL << i;
            const auto tail = 1ULL << (qubit_num - 1 - qubits[N - 1 - i]);
            for (uint64_t j = 0; j < half_cnt; ++j) {
                ret[half_cnt + j] = ret[j] | tail;
            }
        }

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
}

#endif //SIMULATION_BACKENDS_UTILITY_H

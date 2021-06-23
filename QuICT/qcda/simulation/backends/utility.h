//
// Created by Ci Lei on 2021-06-23.
//

#ifndef SIMULATION_BACKENDS_UTILITY_H
#define SIMULATION_BACKENDS_UTILITY_H

#include <cstdint>
#include <array>
#include <complex>
#include <type_traits>

namespace QuICT {
    //https://stackoverflow.com/questions/34672441/stdis-base-of-for-template-classes
    template<template<typename...> class _Base, typename..._Derive_Args>
    std::true_type is_base_of_template_impl(const _Base<_Derive_Args...> *);

    template<template<typename...> class _Base>
    std::false_type is_base_of_template_impl(...);

    template<template<typename...> class _Base, typename _Derived>
    using is_base_of_template = decltype(is_base_of_template_impl<_Base>(std::declval<_Derived *>()));


    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // Data type aliases
    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

    template<uint64_t N>
    using uarray_t = std::array<uint64_t, N>;

    template<typename precision_t>
    using mat_entry_t = std::complex<precision_t>;

    template<typename precision_t, uint64_t N>
    using marray_t = std::array<mat_entry_t<precision_t>, N>;

    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // Helper functions to create indices array
    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

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

    template<uint64_t N>
    inline uarray_t<1ULL << N> index(
            const uint64_t task_id,
            const uint64_t qubit_num,
            const uarray_t<N> qubits,
            const uarray_t<N> qubits_sorted
    ) {
        auto ret = uarray_t<1ULL << N>();
        ret[0] = task_id;

#pragma unroll
        for (uint64_t i = 0; i < N; ++i) {
            uint64_t pos = qubit_num - 1 - qubits_sorted[i];
            uint64_t tail = ret[0] & ((1ULL << pos) - 1);
            ret[0] = ret[0] >> pos << (pos + 1) | tail;
        }

#pragma unroll
        for (uint64_t i = 0; i < N; ++i) {
            const auto half_cnt = 1ULL << i;
            const auto tail = 1ULL << qubits[i];
            for (uint64_t j = 0; j < half_cnt; ++j) {
                ret[half_cnt + j] = ret[j] | tail;
            }
        }

        return ret;
    }


}

#endif //SIMULATION_BACKENDS_UTILITY_H

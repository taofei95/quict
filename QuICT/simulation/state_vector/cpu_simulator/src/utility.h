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
#include <vector>
#include <cassert>
#include <fstream>
#include <thread>
#include <map>
#include <immintrin.h>
#include <omp.h>

//#define DEFAULT_NUM_THREADS 8

// v1 * v2 == (v1r * v2r - v1i * v2i) + (v1i * v2r + v1r * v2i)*J
#define COMPLEX_YMM_MUL(v1r, v1i, v2r, v2i, res_r, res_i) \
    res_r = _mm256_mul_pd(v1r, v2r);\
    res_r = _mm256_fnmadd_pd(v1i, v2i, res_r);\
    res_i = _mm256_mul_pd(v1i, v2r);\
    res_i = _mm256_fmadd_pd(v1r, v2i, res_i);

#define COMPLEX_YMM_NORM(vr, vi, res) \
    res = _mm256_mul_pd(vr, vr);      \
    res = _mm256_fmadd_pd(vi, vi, res);

#define STRIDE_2_LOAD_ODD_PD(from_addr, to_reg, tmp1, tmp2) \
    tmp1 = _mm256_loadu_pd(from_addr);\
    tmp2 = _mm256_loadu_pd(&(((double*)(from_addr))[4]));\
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
        special_x,
        measure
    };

    const std::map<std::string, gate_category> dispatcher = {
            {"special_h",    gate_category::special_h},
            {"special_x",    gate_category::special_x},
            {"diag_1",       gate_category::diag_1},
            {"diag_2",       gate_category::diag_2},
            {"ctrl_diag",    gate_category::ctrl_diag},
            {"unitary_1",    gate_category::unitary_1},
            {"unitary_2",    gate_category::unitary_2},
            {"ctrl_unitary", gate_category::ctrl_unitary},
            {"measure",      gate_category::measure}
    };

    inline uint64_t omp_chunk_size(uint64_t qubit_num, uint64_t batch_size = 4) {
#define MSB(x) (63 - __builtin_clzll(x))
        constexpr uint64_t SCALE_FACTOR = 4;
        constexpr uint64_t MAX_CHUNK_SIZE = 1024;
        constexpr uint64_t MIN_CHUNK_SIZE = 1;

        uint64_t b_thread = MSB(omp_get_thread_num());
        uint64_t task_size = (1 << qubit_num) / batch_size;
        uint64_t chunk_size = std::max(std::min(task_size >> b_thread >> SCALE_FACTOR, MAX_CHUNK_SIZE), MIN_CHUNK_SIZE);
//        std::cout << "chunk_size=" << chunk_size << std::endl;
        return chunk_size;
#undef MSB
    }

    template<typename Precision>
    inline std::pair<Precision *, Precision *> separate_complex(
            uint64_t q_state_bit_num,
            const std::complex<Precision> *c_arr
    ) {
        auto len = 1ULL << q_state_bit_num;
        auto real = new Precision[len];
        auto imag = new Precision[len];
        if (q_state_bit_num >= 2) {
//#pragma omp parallel for num_threads(DEFAULT_NUM_THREADS) schedule(dynamic, omp_chunk_size(q_state_bit_num))
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
        } else {
            real[0] = c_arr[0].real();
            imag[0] = c_arr[0].imag();

            real[1] = c_arr[1].real();
            imag[1] = c_arr[1].imag();
        }
    }

    template<typename Precision>
    inline void combine_complex(
            uint64_t q_state_bit_num,
            const Precision *real,
            const Precision *imag,
            std::complex<Precision> *res
    ) {
        if (q_state_bit_num >= 2) {
            auto len = 1ULL << q_state_bit_num;
//#pragma omp parallel for num_threads(DEFAULT_NUM_THREADS) schedule(dynamic, omp_chunk_size(q_state_bit_num))
            for (uint64_t i = 0; i < len; i += 4) {
                res[i] = {real[i], imag[i]};
                res[i + 1] = {real[i + 1], imag[i + 1]};
                res[i + 2] = {real[i + 2], imag[i + 2]};
                res[i + 3] = {real[i + 3], imag[i + 3]};
            }
        } else {
            res[0] = {real[0], imag[0]};
            res[1] = {real[1], imag[1]};
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
                affect_args_(std::forward<_u64_vec_T>(affect_args)) {}

        template<typename _u64_vec_T>
        GateDescription(
                const char *gate_name,
                _u64_vec_T &&affect_args,
                std::vector<std::complex<Precision>> &&data_ptr
        ) :
                gate_name_(gate_name),
                affect_args_(std::forward<_u64_vec_T>(affect_args)),
                data_ptr_(std::forward<std::vector<std::complex<Precision>>>(data_ptr)) {}
    };

    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // Helper to detect system config
    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

    struct SysConf {
        uint32_t omp_num_thread_ = std::thread::hardware_concurrency();
        uint32_t omp_threshold_ = 10;
    };

    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // Helper functions to create indices array
    //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

    /*
     * Index is created from task_id and qubit positions.
     * Insert some bits to task_id's binary form at positions
     * specified by qubit positions.
     * Example:
     * task_id has binary form of (abcd). And qubits = {0,2}.
     * Result would be {(abc0d0), (abc0d1), (abc1d0), (abc1d1)}.
     * If qubits are {2, 0} instead of {0, 2}, result is {(abc0d0), (abc1d0), (abc0d1), (abc1d1)}.
     * */

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
}

#endif //SIMULATION_BACKENDS_UTILITY_H

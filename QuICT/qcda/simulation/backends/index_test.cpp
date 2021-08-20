//
// Created by Ci Lei on 2021-08-13.
//

#include <gtest/gtest.h>
#include <cstdint>
#include <iostream>
#include <random>
#include <array>

#include "utility.h"

template<
        uint64_t N = 1,
        typename std::enable_if<(N == 1), int>::type dummy = 0
>
void index_test(uint64_t repeat_rounds, uint64_t qubit_num)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> targ_dist(0, qubit_num - 1);
    std::uniform_int_distribution<uint64_t> task_id_dist(0, (1ULL << (qubit_num - 1)) - 1);

    uint64_t tick_cnt = 0; // 1 tick is 1 us
    std::cout << "1 Bit Test" << std::endl;
    for (int i = 0; i < repeat_rounds; ++i) {
        uint64_t targ = targ_dist(gen);
        uint64_t task_id = task_id_dist(gen);
        auto cnt = QuICT::time_elapse(QuICT::index<1>, task_id, qubit_num, targ);
        tick_cnt += cnt;
    }
    double rate = 1.0 * tick_cnt / repeat_rounds;
    std::cout << rate << " us/op" << std::endl;
}

template<
        uint64_t N = 1,
        typename std::enable_if<(N > 1), int>::type dummy = 0
>
void index_test(uint64_t repeat_rounds, uint64_t qubit_num)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> targ_dist(0, qubit_num - 1);
    std::uniform_int_distribution<uint64_t> task_id_dist(0, (1ULL << (qubit_num - 1)) - 1);

    uint64_t tick_cnt = 0; // 1 tick is 1 us
    std::cout << N << " Bit Test" << std::endl;
    for (int i = 0; i < repeat_rounds; ++i) {
        std::array<uint64_t, N> qubits;
        for(int i = 0; i < N; i++) qubits[i] = targ_dist(gen);
        std::array<uint64_t, N> qubits_sorted(qubits);
        std::sort(qubits_sorted.begin(), qubits_sorted.end());

        uint64_t task_id = task_id_dist(gen);
        auto cnt = QuICT::time_elapse(QuICT::index<N>, task_id, qubit_num, qubits, qubits_sorted);
        tick_cnt += cnt;
    }
    double rate = 1.0 * tick_cnt / repeat_rounds;
    std::cout << rate << " us/op" << std::endl;
}


TEST(IndexTest, BaseLineTest) {
    // The index<1> is quite fast. Small repeat values lead inaccurate test results.
//    uint64_t repeat_rounds = 1e8;
//    uint64_t qubit_num = 30;
//    std::random_device rd;
//    std::mt19937 gen(rd());
//    std::uniform_int_distribution<uint64_t> targ_dist(0, qubit_num - 1);
//    std::uniform_int_distribution<uint64_t> task_id_dist(0, (1ULL << (qubit_num - 1)) - 1);
//
//    uint64_t tick_cnt = 0; // 1 tick is 1 us
//    double rate;
//
//    std::cout << "1 Bit Test" << std::endl;
//    for (int i = 0; i < repeat_rounds; ++i) {
//        uint64_t targ = targ_dist(gen);
//        uint64_t task_id = task_id_dist(gen);
//        auto cnt = QuICT::time_elapse(QuICT::index<1>, task_id, qubit_num, targ);
//        tick_cnt += cnt;
//    }
//    rate = 1.0 * tick_cnt / repeat_rounds;
//    std::cout << rate << " us/op" << std::endl;
//
//    tick_cnt = 0;
//    std::cout << "2 Bit Test" << std::endl;
//    for (int i = 0; i < repeat_rounds; ++i) {
//        std::array<uint64_t, 2> qubits = {targ_dist(gen), targ_dist(gen)};
//        std::array<uint64_t, 2> qubits_sorted;
//        if (qubits[0] > qubits[2]) {
//            qubits_sorted[0] = qubits[1];
//            qubits_sorted[1] = qubits[0];
//        } else {
//            qubits_sorted[0] = qubits[0];
//            qubits_sorted[1] = qubits[1];
//        }
//        uint64_t task_id = task_id_dist(gen);
//        auto cnt = QuICT::time_elapse(QuICT::index<2>, task_id, qubit_num, qubits, qubits_sorted);
//        tick_cnt += cnt;
//    }
//    rate = 1.0 * tick_cnt / repeat_rounds;
//    std::cout << rate << " us/op" << std::endl;

//    index_test<1>(1e8, 30);
//    index_test<2>(1e8, 30);

//    index_test<3>(1e6, 30);
//    index_test<5>(1e6, 30);
//    index_test<7>(1e6, 30);

//    index_test<2>(1e8, 30);
    // index_test<1>(1e8, 30);
    index_test<2>(1e8, 30);
}
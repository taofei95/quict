//
// Created by Ci Lei on 2021-07-06.
//

#include <gtest/gtest.h>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <functional>
#include <array>
#include "utility.h"

void nothing_at_all(int x, int y, int z) {
    constexpr int sz = 10000;
    int a[sz];
    std::fill(a, a + sz, 0);
    a[0] = x & y & z;
    for (uint64_t i = 1; i < sz; ++i) {
        a[i] = (a[i - 1] + 1) & (a[i - 1] - 1);
    }
}

TEST(MiscTest, TimeTest) {
    using namespace std;
    using namespace QuICT;
    cout << "[TimeMeasureTemplate]" << endl;
    auto tm = time_elapse(nothing_at_all, 1, 2, 3);
    cout << tm << "[us]" << endl;
}

TEST(MiscTest, SingleIndexTest) {
    using namespace std;

    cout << "[SingleIndexTest]" << endl;
    uint64_t rnd = 1e7;
    int qubit_num = 30;
    uint64_t cnt = 0;
    for (uint64_t i = 0; i < rnd; ++i) {
        cnt += QuICT::time_elapse(QuICT::index<1>, i, qubit_num, i % qubit_num);
    }
    auto tps = 1.0 * cnt / rnd;
    cout << tps << "[us/op]" << endl;
}

TEST(MiscTest, MultiIndexTest) {
    using namespace std;

    cout << "[MultiIndexTest]" << endl;
    uint64_t rnd = 1e7;
    int qubit_num = 30;
    uint64_t cnt = 0;
    for (uint64_t i = 0; i < rnd; ++i) {
        QuICT::uarray_t<2> qubits = {i, (i * i % qubit_num)};
        QuICT::uarray_t<2> qubits_sorted = {i, (i * i % qubit_num)};
        if (qubits_sorted[0] > qubits_sorted[1]) {
            swap(qubits_sorted[0], qubits_sorted[1]);
        }
        cnt += QuICT::time_elapse(QuICT::index<2>, i, qubit_num, qubits, qubits_sorted);
    }
    auto tps = 1.0 * cnt / rnd;
    cout << tps << "[us/op]" << endl;
}
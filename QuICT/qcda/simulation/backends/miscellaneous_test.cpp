//
// Created by Ci Lei on 2021-07-06.
//

#include <gtest/gtest.h>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <functional>
#include <array>
#include <immintrin.h>
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

TEST(MiscTest, SingleIndexPerf) {
    using namespace std;

    cout << "[SingleIndexPerf]" << endl;
    uint64_t rnd = 1e7;
    int qubit_num = 30;
    uint64_t cnt = 0;
    for (uint64_t i = 0; i < rnd; ++i) {
        cnt += QuICT::time_elapse(QuICT::index<1>, i, qubit_num, i % qubit_num);
    }
    auto tps = 1.0 * cnt / rnd;
    cout << tps << "[us/op]" << endl;
}

TEST(MiscTest, MultiIndexPerf) {
    using namespace std;

    cout << "[MultiIndexPerf]" << endl;
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

TEST(MiscTest, SingleIndexTest) {
    using namespace std;

    cout << "[SingleIndexTest]" << endl;
    uint64_t qubit_num = 20;
    uint64_t task_num = 1ULL << (qubit_num - 1);
    uint64_t targ = 3;
    vector<uint64_t> res_vec;
    for (uint64_t task_id = 0; task_id < task_num; ++task_id) {
        auto ind = QuICT::index(task_id, qubit_num, targ);
        res_vec.push_back(ind[0]);
        res_vec.push_back(ind[1]);
    }
    ASSERT_EQ(1ULL << qubit_num, res_vec.size());
    sort(res_vec.begin(), res_vec.end());
    ASSERT_EQ(0, res_vec[0]);
    for (uint64_t i = 1; i < (1ULL << qubit_num); ++i) {
        ASSERT_EQ(res_vec[i], res_vec[i - 1] + 1);
    }
}

TEST(MiscTest, MultiIndexTest) {
    using namespace std;

    cout << "[MultiIndexTest]" << endl;
    uint64_t qubit_num = 20;
    uint64_t task_num = 1ULL << (qubit_num - 2);
    QuICT::uarray_t<2> qubits = {6, 3};
    QuICT::uarray_t<2> qubits_sorted = {3, 6};
    vector<uint64_t> res_vec;
    for (uint64_t task_id = 0; task_id < task_num; ++task_id) {
        auto ind = QuICT::index(task_id, qubit_num, qubits, qubits_sorted);
        auto ind0 = QuICT::index0(task_id, qubit_num, qubits, qubits_sorted);
        ASSERT_EQ(ind0, ind[0]);
        res_vec.push_back(ind[0]);
        res_vec.push_back(ind[1]);
        res_vec.push_back(ind[2]);
        res_vec.push_back(ind[3]);
    }
    uint64_t mask01 = (1ULL << (qubit_num - 1 - qubits[1]));
    uint64_t mask10 = (1ULL << (qubit_num - 1 - qubits[0]));
    uint64_t mask11 = (1ULL << (qubit_num - 1 - qubits[1])) | (1UL << (qubit_num - 1 - qubits[0]));

    ASSERT_EQ(1ULL << qubit_num, res_vec.size());
    ASSERT_EQ(0, res_vec[0]);
    ASSERT_EQ(res_vec[1], mask01);
    ASSERT_EQ(res_vec[2], mask10);
    ASSERT_EQ(res_vec[3], mask11);

    sort(res_vec.begin(), res_vec.end());


    for (uint64_t i = 1; i < (1ULL << qubit_num); ++i) {
        ASSERT_EQ(res_vec[i], res_vec[i - 1] + 1) << "i = " << i;
    }
}

TEST(MiscTest, AVXTest) {
    double arr[4] = {1, 2, 3, 4};
    double res[4];

    __m256d ymm0 = _mm256_loadu2_m128d(&arr[2], &arr[0]);
    _mm256_storeu2_m128d(&res[2], &res[0], ymm0);

    EXPECT_DOUBLE_EQ(arr[0], res[0]);
    EXPECT_DOUBLE_EQ(arr[1], res[1]);
    EXPECT_DOUBLE_EQ(arr[2], res[2]);
    EXPECT_DOUBLE_EQ(arr[3], res[3]);
}

TEST(MiscTest, StridLoadStoreTest) {
    double a[8], a_cpy[8];
    double b[8], b_cpy[8];
    double tmp[4];
    constexpr double eps = 1e-6;
    for (int i = 0; i < 8; ++i) {
        a[i] = a_cpy[i] = i;
        b[i] = b_cpy[i] = -i;
    }
    __m256d ymm0, ymm1, ymm2, ymm3;
    STRIDE_2_LOAD_ODD_PD(a, ymm0, ymm2, ymm3);
    STRIDE_2_LOAD_ODD_PD(a_cpy, ymm1, ymm2, ymm3);
    STRIDE_2_STORE_ODD_PD(b, ymm0, tmp);
    STRIDE_2_STORE_ODD_PD(b_cpy, ymm1, tmp);
    for (int i = 0; i < 8; ++i) {
        if (i & 1) {
            ASSERT_NEAR(b[i], i, eps);
            ASSERT_NEAR(b_cpy[i], i, eps);
        } else {
            ASSERT_NEAR(b[i], -i, eps);
            ASSERT_NEAR(b_cpy[i], -i, eps);
        }
    }
}
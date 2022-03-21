//
// Created by Ci Lei on 2021-10-20.
//

#include <gtest/gtest.h>
#include <random>
#include <chrono>
#include <complex>

#include "utility.h"
#include "q_state.h"

TEST(QStateTest, TestMerge) {
    constexpr double eps = 1e-6;

    std::mt19937 rand_gen((std::random_device())());
    auto dist = std::uniform_real_distribution<double>(-1000, 1000);

    for (int a_len = 1; a_len < 5; a_len += 1) {
        for (int b_len = 1; b_len < 5; b_len += 1) {
            auto a = new QuICT::QState<double>(0, a_len);
            auto b = new QuICT::QState<double>(1, b_len);

            for (int i = 0; i < (1ULL << a->qubit_num_); i += 1) {
                a->real_[i] = dist(rand_gen);
                a->imag_[i] = dist(rand_gen);
            }
            for (int i = 0; i < (1ULL << b->qubit_num_); i += 1) {
                b->real_[i] = dist(rand_gen);
                b->imag_[i] = dist(rand_gen);
            }

            auto res = QuICT::QState<double>::merge_q_state(a, b, 0);
            for (int i = 0; i < (1ULL << a->qubit_num_); i += 1) {
                auto blk_len = 1ULL << b->qubit_num_;
                for (int j = 0; j < blk_len; j += 1) {
                    auto a_c = std::complex<double>(a->real_[i], a->imag_[i]);
                    auto b_c = std::complex<double>(b->real_[j], b->imag_[j]);
                    auto chk = a_c * b_c;
                    ASSERT_NEAR(res->real_[i * blk_len + j], chk.real(), eps) << "i = " << i << "j = " << j;
                    ASSERT_NEAR(res->imag_[i * blk_len + j], chk.imag(), eps) << "i = " << i << "j = " << j;
                }
            }
        }
    }
}

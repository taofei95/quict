//
// Created by Ci Lei on 2021-10-06.
//

#include <gtest/gtest.h>

#include "utility.h"
#include "test_utility.h"
#include "tiny_simulator.h"

auto simulator = QuICT::TinySimulator<double>();

TEST(TinyTest, HTest) {
    for (int i = 1; i <= 4; ++i) {
        std::string category = "./test_data/tiny_h_";
        test_by_data_file(category + std::to_string(i) + ".txt", simulator);
    }
}

TEST(TinyTest, DiagTest) {
    for (int i = 1; i <= 4; ++i) {
        std::string category = "./test_data/tiny_diag_";
        test_by_data_file(category + std::to_string(i) + ".txt", simulator);
    }
}

TEST(TinyTest, XTest) {
    for (int i = 1; i <= 4; ++i) {
        std::string category = "./test_data/tiny_x_";
        test_by_data_file(category + std::to_string(i) + ".txt", simulator);
    }
}

TEST(TinyTest, CtrlDiagTest) {
    for (int i = 2; i <= 4; ++i) {
        std::string category = "./test_data/tiny_ctrl_diag_";
        test_by_data_file(category + std::to_string(i) + ".txt", simulator);
    }
}

TEST(TinyTest, UnitaryTest) {
    for (int i = 1; i <= 4; ++i) {
        std::string category = "./test_data/tiny_unitary_";
        test_by_data_file(category + std::to_string(i) + ".txt", simulator);
    }
}

TEST(TinyTest, CtrlUnitaryTest) {
    for (int i = 2; i <= 4; ++i) {
        std::string category = "./test_data/tiny_ctrl_unitary_";
        test_by_data_file(category + std::to_string(i) + ".txt", simulator);
    }
}


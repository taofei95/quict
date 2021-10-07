//
// Created by Ci Lei on 2021-10-06.
//

#include <gtest/gtest.h>

#include "utility.h"
#include "test_utility.h"
#include "tiny_simulator.h"

auto simulator = QuICT::TinySimulator<double>();

TEST(TinyTest, Diag1Test) {
    for (int i = 1; i <= 4; ++i) {
        std::string category = "tiny_diag_";
        test_by_data_file(category + std::to_string(i) + ".txt", simulator);
    }
}


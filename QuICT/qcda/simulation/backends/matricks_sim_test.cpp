//
// Created by Ci Lei on 2021-07-12.
//

#include <gtest/gtest.h>

#include "utility.h"
#include "test_utility.h"
#include "matricks_simulator.h"


auto simulator = QuICT::MaTricksSimulator<double>();

TEST(MatricksTest, HTest) {
    test_by_data_file("./test_data/h.txt", simulator);
}

TEST(MatricksTest, CtrlDiagTest) {
    test_by_data_file("./test_data/crz.txt", simulator);
}

TEST(MatricksTest, QftTest) {
    test_by_data_file("./test_data/qft.txt", simulator);
}


TEST(MatricksTest, DiagTest) {
    test_by_data_file("./test_data/diag.txt", simulator);
}

TEST(MatricksTest, XTest) {
    test_by_data_file("./test_data/x.txt", simulator);
}

TEST(MatricksTest, CtrlUnitaryTest) {
    test_by_data_file("./test_data/cu.txt", simulator);
}

TEST(MatricksTest, UnitaryTest) {
    test_by_data_file("./test_data/u.txt", simulator);
}
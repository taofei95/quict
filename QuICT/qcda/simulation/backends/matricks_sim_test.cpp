//
// Created by Ci Lei on 2021-07-12.
//

#include <gtest/gtest.h>

#include "utility.h"
#include "test_utility.h"
#include "matricks_simulator.h"


auto simulator = QuICT::MaTricksSimulator<double>();

TEST(MatricksTest, HTest) {
    test_by_data_file("h.txt", simulator);
}

TEST(MatricksTest, CtrlDiagTest) {
    test_by_data_file("crz.txt", simulator);
}

TEST(MatricksTest, QftTest) {
    test_by_data_file("qft.txt", simulator);
}


TEST(MatricksTest, DiagTest) {
    test_by_data_file("diag.txt", simulator);
}

TEST(MatricksTest, XTest) {
    test_by_data_file("x.txt", simulator);
}

TEST(MatricksTest, CtrlUnitaryTest) {
    test_by_data_file("cu.txt", simulator);
}

TEST(MatricksTest, UnitaryTest) {
    test_by_data_file("u.txt", simulator);
}
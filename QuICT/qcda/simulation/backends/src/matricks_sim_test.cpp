//
// Created by Ci Lei on 2021-07-12.
//

#include <gtest/gtest.h>

#include "utility.h"
#include "test_utility.h"
#include "matricks_simulator.h"


auto simulator = QuICT::MaTricksSimulator<double>();

TEST(MatricksTest, HTest) {
    test_stateless_simulator("./test_data/h.txt", simulator);
}

TEST(MatricksTest, CtrlDiagTest) {
    test_stateless_simulator("./test_data/crz.txt", simulator);
}

TEST(MatricksTest, QftTest) {
    test_stateless_simulator("./test_data/qft.txt", simulator);
}


TEST(MatricksTest, Diag1Test) {
    test_stateless_simulator("./test_data/diag.txt", simulator);
}

TEST(MatricksTest, Diag2Test) {
    test_stateless_simulator("./test_data/diag_2.txt", simulator);
}

TEST(MatricksTest, XTest) {
    test_stateless_simulator("./test_data/x.txt", simulator);
}

TEST(MatricksTest, CtrlUnitaryTest) {
    test_stateless_simulator("./test_data/cu.txt", simulator);
}

TEST(MatricksTest, UnitaryTest) {
    test_stateless_simulator("./test_data/u.txt", simulator);
}
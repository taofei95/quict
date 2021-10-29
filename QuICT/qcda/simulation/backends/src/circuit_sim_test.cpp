//
// Created by Ci Lei on 2021-10-20.
//

#include <gtest/gtest.h>

#include "utility.h"
#include "test_utility.h"
#include "q_state.h"
#include "circuit_simulator.h"

TEST(CircuitTest, HTest) {
    test_circuit_simulator("./test_data/h.txt");
}

TEST(CircuitTest, XTest) {
    test_circuit_simulator("./test_data/x.txt");
}

TEST(CircuitTest, Diag1Test) {
    test_circuit_simulator("./test_data/diag.txt");
}

TEST(CircuitTest, Diag2Test) {
    test_circuit_simulator("./test_data/diag_2.txt");
}

TEST(CircuitTest, CtrlDiagTest) {
    test_circuit_simulator("./test_data/crz.txt");
}

TEST(CircuitTest, UnitaryTest) {
    test_circuit_simulator("./test_data/u.txt");
}

TEST(CircuitTest, CtrlUnitaryTest) {
    test_circuit_simulator("./test_data/cu.txt");
}

TEST(CircuitTest, QFTTest) {
    test_circuit_simulator("./test_data/qft.txt");
}

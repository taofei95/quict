//
// Created by Ci Lei on 2021-06-23.
//
#include <iostream>
#include <gtest/gtest.h>
#include <complex>
#include "utility.h"
#include "gate.h"
#include "simulator.h"

using namespace QuICT;

//TEST (RunningChecker, ExpectError) {
//    EXPECT_EQ(1, 2);
//}

TEST(TypeTraisTest, GateQubitNum) {
    auto h_gate = Gate::HGate<double>(0);
    auto z_gate = Gate::ZGate<double>(0);
    auto crz_gate = Gate::CrzGate<double>(0, 1, 0);

    EXPECT_EQ(1, Gate::gate_qubit_num<decltype(h_gate)>::value);
    EXPECT_EQ(1, Gate::gate_qubit_num<decltype(z_gate)>::value);
    EXPECT_EQ(2, Gate::gate_qubit_num<decltype(crz_gate)>::value);
}

TEST(SimTest, RunCheck) {
    auto simulator = Simulator<double>();
    simulator.append_gate("h", 0, {0}, 0);
    auto diagonal = new std::complex<double>[2];
    diagonal[0] = 1, diagonal[1] = -1;
    simulator.append_gate("crz", 0, {0, 1}, diagonal);
    auto init_state = new std::complex<double>[1ULL << 10];
    simulator.run(10, init_state);
}

//TEST(SimTest, RunQFT) {
//    auto simulator = Simulator<double>();
//
//}

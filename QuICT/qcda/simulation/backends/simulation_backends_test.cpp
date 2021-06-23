//
// Created by Ci Lei on 2021-06-23.
//
#include <gtest/gtest.h>
#include "utility.h"
#include "gate.h"

using namespace QuICT;

TEST(TypeTraisTest, GateSubType) {
    auto h_gate = Gate::HGate<double>(0);
    auto z_gate = Gate::ZGate<double>(0);
    auto crz_gate = Gate::CrzGate<double>(0, 1, 0);

    EXPECT_EQ(true, Gate::is_single_bit<decltype(h_gate)>::value);
    EXPECT_EQ(true, Gate::is_single_bit<decltype(z_gate)>::value);
    EXPECT_EQ(true, Gate::is_controlled_diagonal_gate<decltype(crz_gate)>::value);

    EXPECT_EQ(true, Gate::is_controlled_2_bit<decltype(crz_gate)>::value);
    EXPECT_EQ(true, Gate::is_diagonal_gate<decltype(z_gate)>::value);
}

TEST(TypeTraisTest, GateQubitNum) {
    auto h_gate = Gate::HGate<double>(0);
    auto z_gate = Gate::ZGate<double>(0);
    auto crz_gate = Gate::CrzGate<double>(0, 1, 0);

    EXPECT_EQ(1, Gate::gate_qubit_num<decltype(h_gate)>::value);
    EXPECT_EQ(1, Gate::gate_qubit_num<decltype(z_gate)>::value);
    EXPECT_EQ(2, Gate::gate_qubit_num<decltype(crz_gate)>::value);
}

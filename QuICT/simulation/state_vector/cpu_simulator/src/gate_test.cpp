//
// Created by Ci Lei on 2021-08-01.
//

#include <gtest/gtest.h>
#include <type_traits>
#include <iostream>
#include "gate.h"
#include "utility.h"

TEST(GateTest, TraitTest) {
    using namespace QuICT;
    EXPECT_EQ(true, is_simple_gate_v<XGate<double>>);
    EXPECT_EQ(true, is_simple_gate_v<HGate<double>>);
    EXPECT_EQ(true, is_simple_gate_v<XGate<float>>);
    EXPECT_EQ(true, is_simple_gate_v<HGate<float>>);

    EXPECT_EQ(true, is_ctrl_diag_gate_v<CrzGate<double>>);
    EXPECT_EQ(true, is_ctrl_diag_gate_v<CrzGate<float>>);

    EXPECT_EQ(1, gate_qubit_num_v<HGate<double>>);
    EXPECT_EQ(1, gate_qubit_num_v<XGate<double>>);
    EXPECT_EQ(2, gate_qubit_num_v<CrzGate<double>>);
}


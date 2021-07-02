//
// Created by Ci Lei on 2021-06-23.
//
#include <iostream>
#include <gtest/gtest.h>
#include <complex>
#include <sstream>
#include <fstream>
#include <string>
#include <cstdint>
#include "utility.h"
#include "gate.h"
#include "monotune_simulator.h"

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
    auto simulator = MonoTuneSimulator<double>();
    auto diagonal = new std::complex<double>[2];
    diagonal[0] = 1, diagonal[1] = -1;
    simulator.append_gate("h", {0}, 0, 0);
    simulator.append_gate("crz", {0, 1}, 0, diagonal);
    auto state = new std::complex<double>[1ULL << 10];
    simulator.run(10, state);
}

TEST(SimTest, QFTCorrectnessCheck) {
    using namespace std;
    auto simulator = MonoTuneSimulator<double>();

    fstream fs;
    fs.open("qft.txt", ios::in);
    if (!fs) {
        ASSERT_EQ(0, 1) << "Open qft description failed.";
    }
    uint64_t qubit_num;
    fs >> qubit_num;
    string gate_name;
    while (fs >> gate_name) {
        if (gate_name == "__TERM__") {
            break;
        }
        uint64_t carg;
        uint64_t targ;
        double parg;
        if (gate_name == "h") {

            fs >> targ;
            simulator.append_gate("h", {targ}, 0, nullptr);
        } else if (gate_name == "crz") {
            fs >> carg >> targ >> parg;
            simulator.append_gate("crz", {carg, targ}, parg, nullptr);
        }
    }
    auto state = new complex<double>[1ULL << qubit_num];
    auto expect_state = new complex<double>[1ULL << qubit_num];
    fill(state, state + (1ULL << qubit_num), complex<double>(0));
    state[0] = complex<double>(1, 0);

    double re, im;
    char sign, img_label;
    for (uint64_t i = 0; i < (1ULL << qubit_num); ++i) {
        fs >> re >> sign >> im >> img_label;
        expect_state[i] = complex<double>(re, sign == '+' ? im : -im);
    }

    simulator.run(qubit_num, state);

    for (uint64_t i = 0; i < (1ULL << qubit_num); ++i) {
        ASSERT_LE(fabs(state[i].real() - expect_state[i].real()), 1e-7)
                                    << i << " " << state[i].real() << " " << expect_state[i].real();
        ASSERT_LE(fabs(state[i].imag() - expect_state[i].imag()), 1e-7)
                                    << i << " " << state[i].real() << " " << expect_state[i].real();
    }
}

TEST(SimTest, HTest) {

}

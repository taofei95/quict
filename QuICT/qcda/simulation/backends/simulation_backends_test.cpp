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

template<typename precision_t>
void test_by_data_file(const char *data_name, MonoTuneSimulator<precision_t> &simulator) {
    using namespace std;

    cout << "Testing by " << data_name << endl;

    fstream fs;
    fs.open(data_name, ios::in);
    if (!fs) {
        ASSERT_EQ(0, 1) << "Open " << data_name << " description failed.";
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
        } else if (gate_name == "x") {
            fs >> targ;
            simulator.append_gate("x", {targ}, 0, nullptr);
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
                                    << i << " " << state[i].imag() << " " << expect_state[i].imag();
    }
    delete[] state;
    delete[] expect_state;
}

TEST(SimTest, HCorrectnessCheck) {
    using namespace std;
    auto simulator = MonoTuneSimulator<double>();

    test_by_data_file("h.txt", simulator);
}

TEST(SimTest, CrzCorrectnessCheck) {
    using namespace std;
    auto simulator = MonoTuneSimulator<double>();

    test_by_data_file("crz.txt", simulator);
}

TEST(SimTest, QFTCorrectnessCheck) {
    using namespace std;
    auto simulator = MonoTuneSimulator<double>();

    test_by_data_file("qft.txt", simulator);
}



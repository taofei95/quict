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
    auto h_gate = HGate<double>(0);
    auto z_gate = ZGate<double>(0);
    auto crz_gate = CrzGate<double>(0, 1, 0);

    EXPECT_EQ(1, gate_qubit_num<decltype(h_gate)>::value);
    EXPECT_EQ(1, gate_qubit_num<decltype(z_gate)>::value);
    EXPECT_EQ(2, gate_qubit_num<decltype(crz_gate)>::value);
}

template<typename precision_t, QuICT::SimulatorMode sim_mode>
void test_by_data_file(
        const char *data_name,
        MonoTuneSimulator<precision_t, sim_mode> &simulator,
        double eps = 1e-6
) {
    using namespace std;

    cout << simulator.name() << " " << "Testing by " << data_name << endl;

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
        ASSERT_LE(fabs(state[i].real() - expect_state[i].real()), eps)
                                    << i << " " << state[i].real() << " " << expect_state[i].real();
        ASSERT_LE(fabs(state[i].imag() - expect_state[i].imag()), eps)
                                    << i << " " << state[i].imag() << " " << expect_state[i].imag();
    }
    delete[] state;
    delete[] expect_state;
}

inline void test_multi_simulator_by_file(const char *data_name) {
    using namespace std;
    using namespace QuICT;
    auto single_simulator = MonoTuneSimulator<double, SimulatorMode::single>();
    auto batch_simulator = MonoTuneSimulator<double, SimulatorMode::batch>();
    auto avx_simulator = MonoTuneSimulator<double, SimulatorMode::avx>();
    auto fma_simulator = MonoTuneSimulator<double, SimulatorMode::fma>();

    test_by_data_file(data_name, single_simulator);
    test_by_data_file(data_name, batch_simulator);
    test_by_data_file(data_name, avx_simulator);
    test_by_data_file(data_name, fma_simulator);

    cout << endl;
}

TEST(SimTest, HCorrectnessCheck) {
    test_multi_simulator_by_file("h.txt");
}

TEST(SimTest, CrzCorrectnessCheck) {
    test_multi_simulator_by_file("crz.txt");
}

TEST(SimTest, QFTCorrectnessCheck) {
    test_multi_simulator_by_file("qft.txt");
}



//
// Created by Ci Lei on 2021-07-12.
//

#include <gtest/gtest.h>
#include <cstdint>
#include <random>
#include <complex>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

#include "hybrid_simulator.h"
#include "utility.h"

auto dist = std::uniform_real_distribution<double>(-1, 1);
std::mt19937 rd;


template<typename Precision>
void test_by_data_file(
        const char *data_name,
        QuICT::HybridSimulator<Precision> &simulator,
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
    std::vector<QuICT::GateDescription<Precision>> gate_desc_vec;
    while (fs >> gate_name) {
        if (gate_name == "__TERM__") {
            break;
        }
        uint64_t carg;
        uint64_t targ;
        double parg;

        if (gate_name == "h") {

            fs >> targ;
            gate_desc_vec.emplace_back(
                    "h",
                    std::vector<uint64_t>{targ},
                    0,
                    nullptr
            );
        } else if (gate_name == "x") {
            fs >> targ;
            gate_desc_vec.emplace_back(
                    "x",
                    std::vector<uint64_t>{targ},
                    0,
                    nullptr
            );
        } else if (gate_name == "crz") {
            fs >> carg >> targ >> parg;
            gate_desc_vec.emplace_back(
                    "crz",
                    std::vector<uint64_t>{carg, targ},
                    parg,
                    nullptr
            );
        } else if (gate_name == "x") {
            fs >> targ;
            gate_desc_vec.emplace_back(
                    "x",
                    std::vector<uint64_t>{targ},
                    0,
                    nullptr
            );
        } else if (gate_name == "u1") {
            fs >> targ;
            auto *mat_ = new complex<Precision>[4];
            for (int i = 0; i < 4; i++) {
                double re, im;
                char sign, img_label;
                fs >> re >> sign >> im >> img_label;
                mat_[i] = complex<double>(re, sign == '+' ? im : -im);
            }

            gate_desc_vec.emplace_back(
                    "u1",
                    std::vector<uint64_t>{targ},
                    0,
                    mat_
            );
        }
    }

    auto expect_state = new complex<double>[1ULL << qubit_num];

    double re, im;
    char sign, img_label;
    for (uint64_t i = 0; i < (1ULL << qubit_num); ++i) {
        fs >> re >> sign >> im >> img_label;
        expect_state[i] = complex<double>(re, sign == '+' ? im : -im);
    }

    std::complex<Precision> *state = simulator.run(qubit_num, gate_desc_vec);
    uint64_t err_cnt = 0;
    for (uint64_t i = 0; i < (1ULL << qubit_num); ++i) {
        EXPECT_LE(fabs(state[i].real() - expect_state[i].real()), eps)
                            << i << " " << state[i].real() << " " << expect_state[i].real();
        EXPECT_LE(fabs(state[i].imag() - expect_state[i].imag()), eps)
                            << i << " " << state[i].imag() << " " << expect_state[i].imag();
        if (fabs(state[i].real() - expect_state[i].real()) > eps ||
            fabs(state[i].imag() - expect_state[i].imag()) > eps) {
            err_cnt += 1;
        }
    }
    std::cout << "Error Count: " << err_cnt << "/" << (1ULL << qubit_num) << std::endl;
    delete[] state;
    delete[] expect_state;
}

auto simulator = QuICT::HybridSimulator<double>();

TEST(HybridTest, HTest) {
    test_by_data_file("h.txt", simulator);
}

TEST(HybridTest, CrzTest) {
    test_by_data_file("crz.txt", simulator);
}

//TEST(HybridTest, QftTest) {
//    test_by_data_file("qft.txt", simulator);
//}
//
//TEST(HybridTest, XTest) {
//    test_by_data_file("x.txt", simulator);
//}

//TEST(HybridTest, U1Test) {
//    test_by_data_file("u1.txt", simulator);
//}
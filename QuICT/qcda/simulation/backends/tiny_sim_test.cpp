//
// Created by Ci Lei on 2021-10-06.
//

#include <gtest/gtest.h>
#include <cstdint>
#include <random>
#include <complex>
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>

#include "tiny_simulator.h"

template<typename _Str_T>
void test_by_data_file(
        _Str_T data_name,
        QuICT::TinySimulator<double> &simulator,
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
    std::vector<QuICT::GateDescription<double>> gate_desc_vec;
    while (fs >> gate_name) {
        if (gate_name == "__TERM__") {
            break;
        }
        uint64_t carg;
        uint64_t targ;

        if (gate_name == "special_h") {
            fs >> targ;
            gate_desc_vec.emplace_back(
                    "special_h",
                    std::vector<uint64_t>{targ}
            );
        } else if (gate_name == "special_x") {
            fs >> targ;
            gate_desc_vec.emplace_back(
                    "special_x",
                    std::vector<uint64_t>{targ}
            );
        } else if (gate_name == "unitary_1") {
            fs >> targ;
            auto mat = std::shared_ptr<std::complex<double>[]>(new complex<double>[4]);
            for (int i = 0; i < 4; i++) {
                double re, im;
                char sign, img_label;
                fs >> re >> sign >> im >> img_label;
                mat[i] = complex<double>(re, sign == '+' ? im : -im);
            }

            gate_desc_vec.emplace_back(
                    "unitary_1",
                    std::vector<uint64_t>{targ},
                    mat
            );
        } else if (gate_name == "unitary_2") {
            fs >> carg >> targ;
            auto mat = std::shared_ptr<std::complex<double>[]>(new complex<double>[16]);
            for (int i = 0; i < 16; i++) {
                double re, im;
                char sign, img_label;
                fs >> re >> sign >> im >> img_label;
                mat[i] = complex<double>(re, sign == '+' ? im : -im);
            }

            gate_desc_vec.emplace_back(
                    "unitary_2",
                    std::vector<uint64_t>{carg, targ},
                    mat
            );
        } else if (gate_name == "diag_1") {
            fs >> targ;
            auto diag = std::shared_ptr<std::complex<double>[]>(new complex<double>[2]);
            for (int i = 0; i < 2; ++i) {
                double re, im;
                char sign, img_label;
                fs >> re >> sign >> im >> img_label;
                diag[i] = complex<double>(re, sign == '+' ? im : -im);
            }
            gate_desc_vec.emplace_back(
                    "diag_1",
                    std::vector<uint64_t>{targ},
                    diag
            );
        } else if (gate_name == "ctrl_diag") {
            fs >> carg >> targ;
            auto diag = std::shared_ptr<std::complex<double>[]>(new complex<double>[2]);
            for (int i = 0; i < 2; ++i) {
                double re, im;
                char sign, img_label;
                fs >> re >> sign >> im >> img_label;
                diag[i] = complex<double>(re, sign == '+' ? im : -im);
            }
            gate_desc_vec.emplace_back(
                    "ctrl_diag",
                    std::vector<uint64_t>{carg, targ},
                    diag
            );
        } else if (gate_name == "ctrl_unitary") {
            fs >> carg >> targ;
            auto mat = std::shared_ptr<std::complex<double>[]>(new complex<double>[4]);
            for (int i = 0; i < 4; i++) {
                double re, im;
                char sign, img_label;
                fs >> re >> sign >> im >> img_label;
                mat[i] = complex<double>(re, sign == '+' ? im : -im);
            }
            gate_desc_vec.emplace_back(
                    "ctrl_unitary",
                    std::vector<uint64_t>{carg, targ},
                    mat
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

    std::complex<double> *state = simulator.run(qubit_num, gate_desc_vec);
    for (uint64_t i = 0; i < (1ULL << qubit_num); ++i) {
        ASSERT_NEAR(state[i].real(), expect_state[i].real(), eps) << "i = " << i;
        ASSERT_NEAR(state[i].imag(), expect_state[i].imag(), eps) << "i = " << i;
    }
    delete[] state;
    delete[] expect_state;
}

auto simulator = QuICT::TinySimulator<double>();

TEST(TinyTest, Diag1Test) {
    for (int i = 1; i <= 4; ++i) {
        std::string category = "tiny_diag_";
        test_by_data_file(category + std::to_string(i) + ".txt", simulator);
    }
}


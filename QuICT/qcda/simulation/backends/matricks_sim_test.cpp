//
// Created by Ci Lei on 2021-07-12.
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

#include "utility.h"
#include "matricks_simulator.h"


template<typename Precision>
void test_by_data_file(
        const char *data_name,
        QuICT::MaTricksSimulator<Precision> &simulator,
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
            auto mat = std::shared_ptr<std::complex<Precision>[]>(new complex<Precision>[4]);
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
            auto mat = std::shared_ptr<std::complex<Precision>[]>(new complex<Precision>[16]);
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
            auto diag = std::shared_ptr<std::complex<Precision>[]>(new complex<Precision>[2]);
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
            auto diag = std::shared_ptr<std::complex<Precision>[]>(new complex<Precision>[2]);
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
            auto mat = std::shared_ptr<std::complex<Precision>[]>(new complex<Precision>[4]);
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

    std::complex<Precision> *state = simulator.run(qubit_num, gate_desc_vec);
    for (uint64_t i = 0; i < (1ULL << qubit_num); ++i) {
        ASSERT_NEAR(state[i].real(), expect_state[i].real(), eps) << "i = " << i;
        ASSERT_NEAR(state[i].imag(), expect_state[i].imag(), eps) << "i = " << i;
    }
    delete[] state;
    delete[] expect_state;
}

auto simulator = QuICT::MaTricksSimulator<double>();

TEST(MatricksTest, HTest) {
    test_by_data_file("h.txt", simulator);
}

TEST(MatricksTest, CtrlDiagTest) {
    test_by_data_file("crz.txt", simulator);
}

TEST(MatricksTest, QftTest) {
    test_by_data_file("qft.txt", simulator);
}


TEST(MatricksTest, DiagTest) {
    test_by_data_file("diag.txt", simulator);
}

TEST(MatricksTest, XTest) {
    test_by_data_file("x.txt", simulator);
}

TEST(MatricksTest, CtrlUnitaryTest) {
    test_by_data_file("cu3.txt", simulator);
}

TEST(MatricksTest, UnitaryTest) {
    test_by_data_file("u.txt", simulator);
}
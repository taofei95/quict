//
// Created by Ci Lei on 2021-10-07.
//

#ifndef SIM_BACK_TEST_UTILITY_H
#define SIM_BACK_TEST_UTILITY_H

#include <gtest/gtest.h>
#include <complex>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include "utility.h"
#include "circuit_simulator.h"

//* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// Test helper
//* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

template<
        typename _str_T,
        typename Precision
>
void get_compare_data(
        _str_T data_name,
        uint64_t &qubit_num,
        std::complex<Precision> **expect_state,
        std::vector<QuICT::GateDescription<Precision>> &gate_desc_vec
) {
    using namespace std;

    fstream fs;
    fs.open(data_name, ios::in);
    if (!fs) {
        ASSERT_EQ(0, 1) << "Open " << data_name << " description failed.";
    }
    fs >> qubit_num;
    string gate_name;
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
            auto mat = std::vector<std::complex<Precision>>(4);
            for (int i = 0; i < 4; i++) {
                double re, im;
                char sign, img_label;
                fs >> re >> sign >> im >> img_label;
                mat[i] = std::complex<Precision>(re, sign == '+' ? im : -im);
            }

            gate_desc_vec.emplace_back(
                    "unitary_1",
                    std::vector<uint64_t>{targ},
                    std::move(mat)
            );
        } else if (gate_name == "unitary_2") {
            fs >> carg >> targ;
            auto mat = std::vector<std::complex<Precision>>(16);
            for (int i = 0; i < 16; i++) {
                double re, im;
                char sign, img_label;
                fs >> re >> sign >> im >> img_label;
                mat[i] = std::complex<Precision>(re, sign == '+' ? im : -im);
            }

            gate_desc_vec.emplace_back(
                    "unitary_2",
                    std::vector<uint64_t>{carg, targ},
                    std::move(mat)
            );
        } else if (gate_name == "ctrl_unitary") {
            fs >> carg >> targ;
            auto mat = std::vector<std::complex<Precision>>(4);
            for (int i = 0; i < 4; i++) {
                double re, im;
                char sign, img_label;
                fs >> re >> sign >> im >> img_label;
                mat[i] = std::complex<Precision>(re, sign == '+' ? im : -im);
            }
            gate_desc_vec.emplace_back(
                    "ctrl_unitary",
                    std::vector<uint64_t>{carg, targ},
                    std::move(mat)
            );
        } else if (gate_name == "diag_1") {
            fs >> targ;
            auto diag = std::vector<std::complex<Precision>>(2);
            for (int i = 0; i < 2; ++i) {
                double re, im;
                char sign, img_label;
                fs >> re >> sign >> im >> img_label;
                diag[i] = std::complex<Precision>(re, sign == '+' ? im : -im);
            }
            gate_desc_vec.emplace_back(
                    "diag_1",
                    std::vector<uint64_t>{targ},
                    std::move(diag)
            );
        } else if (gate_name == "ctrl_diag") {
            fs >> carg >> targ;
            auto diag = std::vector<std::complex<Precision>>(2);
            for (int i = 0; i < 2; ++i) {
                double re, im;
                char sign, img_label;
                fs >> re >> sign >> im >> img_label;
                diag[i] = std::complex<Precision>(re, sign == '+' ? im : -im);
            }
            gate_desc_vec.emplace_back(
                    "ctrl_diag",
                    std::vector<uint64_t>{carg, targ},
                    std::move(diag)
            );
        }
    }

    *expect_state = new std::complex<Precision>[1ULL << qubit_num];

    double re, im;
    char sign, img_label;
    for (uint64_t i = 0; i < (1ULL << qubit_num); ++i) {
        fs >> re >> sign >> im >> img_label;
        (*expect_state)[i] = std::complex<Precision>(re, sign == '+' ? im : -im);
    }
}

template<
        template<typename...> class _sim_T,
        typename _str_T,
        typename Precision
>
void test_simulator(
        _str_T data_name,
        _sim_T<Precision> &simulator,
        double eps = 1e-6
) {
    using namespace std;

    cout << simulator.name() << " " << "Testing by " << data_name << endl;

    uint64_t qubit_num;
    std::vector<QuICT::GateDescription<Precision>> gate_desc_vec;
    complex<Precision> *expect_state;

    get_compare_data(data_name, qubit_num, &expect_state, gate_desc_vec);

    std::complex<Precision> *state;
    if constexpr(is_same_v<_sim_T<Precision>, QuICT::CircuitSimulator<Precision>>) {
        state = simulator.run(gate_desc_vec);
    } else {
        state = simulator.run(qubit_num, gate_desc_vec);
    }
    for (uint64_t i = 0; i < (1ULL << qubit_num); ++i) {
        ASSERT_NEAR(state[i].real(), expect_state[i].real(), eps) << "i = " << i;
        ASSERT_NEAR(state[i].imag(), expect_state[i].imag(), eps) << "i = " << i;
    }
    delete[] state;
    delete[] expect_state;
}


#endif //SIM_BACK_TEST_UTILITY_H

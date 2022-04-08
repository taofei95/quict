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
        std::vector<QuICT::GateDescription<Precision>> &gate_desc_vec,
        bool have_compare_data = true
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
        } else if (gate_name == "diag_2") {
            fs >> carg >> targ;
            auto diag = std::vector<std::complex<Precision>>(4);
            for (int i = 0; i < 4; ++i) {
                double re, im;
                char sign, img_label;
                fs >> re >> sign >> im >> img_label;
                diag[i] = std::complex<Precision>(re, sign == '+' ? im : -im);
            }
            gate_desc_vec.emplace_back(
                    "diag_2",
                    std::vector<uint64_t>{carg, targ},
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
        } else if (gate_name == "measure") {
            fs >> targ;
            gate_desc_vec.emplace_back(
                    "measure",
                    std::vector<uint64_t>{targ}
            );
        } else {
            throw std::runtime_error("Not recognized for " + gate_name + " in " + std::string(__func__));
        }
    }

    *expect_state = nullptr;
    if (have_compare_data) {
        *expect_state = new std::complex<Precision>[1ULL << qubit_num];

        double re, im;
        char sign, img_label;
        for (uint64_t i = 0; i < (1ULL << qubit_num); ++i) {
            fs >> re >> sign >> im >> img_label;
            (*expect_state)[i] = std::complex<Precision>(re, sign == '+' ? im : -im);
        }
    }
}

template<
        template<typename...> class _sim_T,
        typename _str_T,
        typename Precision
>
void test_stateless_simulator(
        _str_T data_name,
        _sim_T<Precision> &simulator,
        double eps = 1e-6,
        bool have_compare_data = true,
        bool have_measure_gate = false,
        int n_run = 1
) {
    using namespace std;

    cout << simulator.name() << " " << "Testing by " << data_name << endl;

    uint64_t qubit_num;
    std::vector<QuICT::GateDescription<Precision>> gate_desc_vec;
    complex<Precision> *expect_state;

    get_compare_data(data_name, qubit_num, &expect_state, gate_desc_vec, have_compare_data);

    std::complex<Precision> *state;
    if (have_measure_gate) {
        state = new complex<Precision>[1 << qubit_num];
        fill(state, state + (1 << qubit_num), 0);
        for (int _ = 0; _ < n_run; _++) {
            std::vector<int> measure_res;
            bool keep_state = false;
            std::complex<Precision> *cur = simulator.run(qubit_num, gate_desc_vec, measure_res, keep_state);
            for (uint64_t i = 0; i < (1 << qubit_num); i++) state[i] += cur[i];
            delete[] cur;
        }
        for (uint64_t i = 0; i < (1 << qubit_num); i++) {
            state[i] /= n_run;
//            cout << state[i] << endl;
        }
    } else {
        std::vector<int> measure_res;
        bool keep_state = false;
        auto start = std::chrono::system_clock::now();
        state = simulator.run(qubit_num, gate_desc_vec, measure_res, keep_state);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Simulation costs " << diff.count() << " s\n";
    }

    if (have_compare_data) {
        for (uint64_t i = 0; i < (1ULL << qubit_num); ++i) {
            ASSERT_NEAR(state[i].real(), expect_state[i].real(), eps) << "i = " << i;
            ASSERT_NEAR(state[i].imag(), expect_state[i].imag(), eps) << "i = " << i;
        }
    } else {
        for (uint64_t i = 0; i < (1ULL << qubit_num); ++i) {
            cout << i << ":\t" << state[i] << endl;
        }
    }

    delete[] state;
    delete[] expect_state;
}

template<
        typename _str_T,
        typename Precision=double
>
void test_circuit_simulator(
        _str_T data_name,
        double eps = 1e-6,
        bool have_compare_data = true,
        bool have_measure_gate = false,
        int n_run = 1
) {
    using namespace std;

    uint64_t qubit_num;
    std::vector<QuICT::GateDescription<Precision>> gate_desc_vec;
    complex<Precision> *expect_state;
    get_compare_data(data_name, qubit_num, &expect_state, gate_desc_vec, have_compare_data);

    auto simulator = QuICT::CircuitSimulator<Precision>();
    cout << simulator.name() << " " << "Testing by " << data_name << endl;

    std::complex<Precision> *state;
    if (have_measure_gate) {
        state = new complex<Precision>[1 << qubit_num];
        fill(state, state + (1 << qubit_num), 0);
        for (int _ = 0; _ < n_run; _++) {
            // FIXME: it seems that CircuitSimulator outputs wrong state values if run multiple times
            auto sim = QuICT::CircuitSimulator<Precision>();
            std::vector<int> measure_res;
            bool keep_state = false;
            std::complex<Precision> *cur = sim.run(qubit_num, gate_desc_vec, measure_res, keep_state);
            for (uint64_t i = 0; i < (1 << qubit_num); i++) state[i] += cur[i];
            delete[] cur;
        }
        for (uint64_t i = 0; i < (1 << qubit_num); i++) {
            state[i] /= n_run;
//            cout << state[i] << endl;
        }
    } else {
        auto sim = QuICT::CircuitSimulator<Precision>();
        std::vector<int> measure_res;
        bool keep_state = false;
        auto start = std::chrono::system_clock::now();
        state = simulator.run(qubit_num, gate_desc_vec, measure_res, keep_state);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Simulation costs " << diff.count() << " s\n";
    }

    if (have_compare_data) {
        for (uint64_t i = 0; i < (1ULL << qubit_num); ++i) {
            ASSERT_NEAR(state[i].real(), expect_state[i].real(), eps) << "i = " << i;
            ASSERT_NEAR(state[i].imag(), expect_state[i].imag(), eps) << "i = " << i;
        }
    } else {
        for (uint64_t i = 0; i < (1ULL << qubit_num); ++i) {
            cout << i << ":\t" << state[i] << endl;
        }
    }

    delete[] state;
    delete[] expect_state;
}


#endif //SIM_BACK_TEST_UTILITY_H

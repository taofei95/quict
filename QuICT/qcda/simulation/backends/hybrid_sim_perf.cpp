//
// Created by Ci Lei on 2021-08-16.
//

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <utility>

#include "gate.h"
#include "hybrid_simulator.h"
#include "utility.h"


template<typename Precision>
void exec_sim(
        QuICT::HybridSimulator<Precision> &simulator,
        Precision **real,
        Precision **imag,
        uint64_t qubit_num,
        const std::vector<QuICT::GateDescription<Precision>> &gate_desc_vec
) {
    auto res = simulator.run_without_combine(qubit_num, gate_desc_vec);
    *real = res.first;
    *imag = res.second;
}

template<typename Precision>
void get_desc_vec_by_file(
        const char *data_name,
        QuICT::HybridSimulator<Precision> &simulator,
        std::vector<QuICT::GateDescription<Precision>> &gate_desc_vec,
        uint64_t &qubit_num
) {
    using namespace std;

    cout << simulator.name() << " " << "Testing by " << data_name << endl;

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
        double parg;

        if (gate_name == "h") {

            fs >> targ;
            gate_desc_vec.emplace_back(
                    "h",
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
                    std::vector<uint64_t>(targ),
                    0,
                    nullptr
            );
        }
    }
}

template<typename Precision>
void test_by_data_file(
        const char *data_name,
        QuICT::HybridSimulator<Precision> &simulator
) {
    using namespace std;
    uint64_t qubit_num;
    std::vector<QuICT::GateDescription<Precision>> gate_desc_vec;

    uint64_t cnt;
    cnt = QuICT::time_elapse(get_desc_vec_by_file<double>, data_name, simulator, gate_desc_vec, qubit_num);
    std::cout << simulator.name() << " test results:" << std::endl;
    std::cout << "Qubit number: " << qubit_num << std::endl;
    std::cout << "Gate description build time: " << cnt / 1000 << " ms" << std::endl;

    Precision *real, *imag;
    cnt = QuICT::time_elapse(exec_sim<double>, simulator, &real, &imag, qubit_num, gate_desc_vec);
    std::cout << "Simulation time: " << cnt / 1000 << " ms" << std::endl;
    delete[] real;
    delete[] imag;
}

TEST(HybridTest, QFTPerf) {
    auto simulator = QuICT::HybridSimulator<double>();
    test_by_data_file("qft_perf.txt", simulator);
}

TEST(HybridTest, HPerf) {
    auto simulator = QuICT::HybridSimulator<double>();
    test_by_data_file("h_perf.txt", simulator);
}
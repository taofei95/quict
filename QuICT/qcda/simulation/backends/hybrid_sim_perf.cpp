//
// Created by Ci Lei on 2021-08-16.
//

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>

#include "gate.h"
#include "hybrid_simulator.h"
#include "utility.h"


template<typename Precision>
void exec_sim(
        QuICT::HybridSimulator<Precision> &simulator,
        std::complex<Precision> **state,
        uint64_t qubit_num,
        const std::vector<QuICT::GateDescription<Precision>> &gate_desc_vec
) {
    *state = simulator.run(qubit_num, gate_desc_vec);
}


template<typename Precision>
void test_by_data_file(
        const char *data_name,
        QuICT::HybridSimulator<Precision> &simulator
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

    std::complex<Precision> *state;
    auto cnt = QuICT::time_elapse(exec_sim<double>, simulator, &state, qubit_num, gate_desc_vec);
    std::cout << simulator.name() << " test results:" << std::endl;
    std::cout << "Qubit number: " << qubit_num << std::endl;
    std::cout << cnt << " us" << std::endl;
    delete[] state;
}

TEST(HybridTest, QFTPerf) {
    auto simulator = QuICT::HybridSimulator<double>();
    test_by_data_file("qft_perf.txt", simulator);
}
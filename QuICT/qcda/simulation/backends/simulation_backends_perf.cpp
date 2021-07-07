//
// Created by Ci Lei on 2021-07-01.
//

#include <gtest/gtest.h>
#include <complex>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include "gate.h"
#include "utility.h"
#include "monotune_simulator.h"

TEST(RunCheck, Ignored) {
    EXPECT_EQ(0, 0);
}

template<typename precision_t>
inline std::vector<QuICT::GateDescription<precision_t>> get_qft_desc(
        uint64_t &qubit_num
) {
    using namespace std;
    fstream fs;
    fs.open("qft.txt", ios::in);
    if (!fs) {
        throw runtime_error("Failed to open qft description.");
    }
    auto vec = std::vector<QuICT::GateDescription<precision_t>>();
    fs >> qubit_num;
    string gate_name;
    while (fs >> gate_name) {
        if (gate_name == "__TERM__") {
            break;
        }
        uint64_t carg;
        uint64_t targ;
        precision_t parg;
        if (gate_name == "h") {
            fs >> targ;
            vec.emplace_back(QuICT::GateDescription<precision_t>("h", {targ}, 0, nullptr));
        } else if (gate_name == "crz") {
            fs >> carg >> targ >> parg;
            vec.emplace_back(QuICT::GateDescription<precision_t>("crz", {carg, targ}, parg, nullptr));
        }
    }
    return vec;
}

template<QuICT::SimulatorMode sim_mode, class precision_t>
inline void perf_simulator(
        QuICT::MonoTuneSimulator<precision_t, sim_mode> &simulator,
        uint64_t qubit_num,
        std::vector<QuICT::GateDescription<precision_t>> desc_vec,
        std::complex<precision_t> *state,
        uint64_t round = 5
) {
    using namespace std;
    using namespace chrono;


    for (const auto &gate_desc:desc_vec) {
        simulator.append_gate(gate_desc);
    }

    steady_clock::time_point start_time, end_time;
    fill(state, state + (1ULL << qubit_num), complex<precision_t>(0));
    state[0] = complex<precision_t>(1);
    uint64_t tm = 0;


    for (uint64_t i = 0; i < round; ++i) {
        start_time = steady_clock::now();
        simulator.run(qubit_num, state);
        end_time = steady_clock::now();
        tm += duration_cast<microseconds>(end_time - start_time).count();
    }
    tm /= round;

    cout << simulator.name() << "\t" << tm << "[us]" << endl;

    for (uint64_t i = 0; i < 10; ++i){
        cout << state[i] << endl;
    }
}

template<class precision_t>
inline void perf_all_simulator(
        uint64_t qubit_num,
        std::vector<QuICT::GateDescription<precision_t>> desc_vec,
        std::complex<precision_t> *state
) {
    using namespace QuICT;
    using namespace std;

    cout << "Qubit = " << qubit_num << endl;

    auto single_simulator = MonoTuneSimulator<double, SimulatorMode::single>();
    auto batch_simulator = MonoTuneSimulator<double, SimulatorMode::batch>();
    auto avx_simulator = MonoTuneSimulator<double, SimulatorMode::avx>();
    auto fma_simulator = MonoTuneSimulator<double, SimulatorMode::fma>();


    // single mode
    perf_simulator(single_simulator, qubit_num, desc_vec, state);

    // batch mode
    perf_simulator(batch_simulator, qubit_num, desc_vec, state);

    // fma_mode
    perf_simulator(fma_simulator, qubit_num, desc_vec, state);

    // avx mode
    perf_simulator(avx_simulator, qubit_num, desc_vec, state);
}

TEST(SimPerf, ModePerfTest) {
    using namespace std;
    using namespace QuICT;
    using namespace chrono;
    cout << "[ModePerfTest]" << endl;
    uint64_t qubit_num;
    auto desc_vec = get_qft_desc<double>(qubit_num);
    auto state = new complex<double>[1ULL << qubit_num];

    perf_all_simulator(qubit_num, desc_vec, state);
}

TEST(SimPerf, HPerfTest) {
    using namespace std;
    using namespace QuICT;
    using namespace chrono;
    cout << "[HPerfTest]" << endl;
    uint64_t qubit_num = 20;
    auto desc_vec = vector<GateDescription<double>>();
    for (uint64_t i = 0; i < 500; ++i) {
        desc_vec.emplace_back(GateDescription<double>("h", {i % qubit_num}, 0, nullptr));
    }
    auto state = new complex<double>[1ULL << qubit_num];

    perf_all_simulator(qubit_num, desc_vec, state);
}

TEST(SimPerf, CrzPerfTest) {
    using namespace std;
    using namespace QuICT;
    using namespace chrono;
    cout << "[CrzPerfTest]" << endl;
    uint64_t qubit_num = 20;
    auto desc_vec = vector<GateDescription<double>>();
    for (uint64_t i = 0; i < 500; ++i) {
        desc_vec.emplace_back(GateDescription<double>(
                "crz", {i % qubit_num, (i + 1) % qubit_num}, i, nullptr));
    }
    auto state = new complex<double>[1ULL << qubit_num];

    perf_all_simulator(qubit_num, desc_vec, state);
}
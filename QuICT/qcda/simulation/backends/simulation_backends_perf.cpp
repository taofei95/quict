//
// Created by Ci Lei on 2021-07-01.
//

#include <gtest/gtest.h>
#include <complex>
#include <cstdint>
#include <chrono>
#include <iostream>
#include "gate.h"
#include "utility.h"
#include "simulator.h"

TEST(RunCheck, Ignored) {
    EXPECT_EQ(0, 0);
}

template<QuICT::SimulatorMode sim_mode, class precision_t>
inline void perf_simulator(
        QuICT::Simulator<precision_t, sim_mode> &simulator,
        uint64_t qubit_num,
        std::complex<precision_t> *state,
        uint64_t round = 20
) {
    using namespace std;
    using namespace chrono;


    steady_clock::time_point start_time, end_time;
    uint64_t cnt = 0;

    for (uint64_t i = 0; i < round; ++i) {
        fill(state, state + (1ULL << qubit_num), complex<double>(0));
        state[0] = complex<double>(1);
        start_time = steady_clock::now();
        simulator.run(qubit_num, state);
        end_time = steady_clock::now();
        cnt += duration_cast<microseconds>(end_time - start_time).count();
    }
    cout << cnt * 1.0 / round << "[us]" << endl;
}

TEST(SimPerf, ModePerfTest) {
    using namespace std;
    using namespace QuICT;
    using namespace chrono;
    uint64_t qubit_num = 25;
    auto state = new complex<double>[1ULL << qubit_num];

    auto single_simulator = Simulator<double, SimulatorMode::single>();
    auto batch_simulator = Simulator<double, SimulatorMode::batch>();


    // single mode
    perf_simulator(single_simulator, qubit_num, state);

    // batch mode
    perf_simulator(batch_simulator, qubit_num, state);
}
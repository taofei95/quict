#ifndef QUICT_SIM_BACKEND_BENCH_QUICT_SIM_H
#define QUICT_SIM_BACKEND_BENCH_QUICT_SIM_H

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "../gate/gate.hpp"
#include "../simulator/simulator.hpp"
#include "../utility/complex_detect.hpp"

namespace bench {

namespace detail {

static std::random_device rd;
static std::mt19937 gen(rd());

template <typename T>
inline std::enable_if_t<std::is_integral_v<T>, T> RandNum() {
  static std::uniform_int_distribution<T> dist;
  return dist(gen);
}

template <typename T>
inline std::enable_if_t<std::is_floating_point_v<T>, T> RandNum() {
  static std::uniform_real_distribution<T> dist;
  return dist(gen);
}

template <typename Complex>
inline Complex RandComplex() {
  if constexpr (util::complex_is_f32_v<Complex>) {
    return {RandNum<float>(), RandNum<float>()};
  } else if constexpr (util::complex_is_f64_v<Complex>) {
    return {RandNum<double>(), RandNum<double>()};
  } else {
    // Always fail.
    static_assert(!sizeof(Complex),
                  "Not support complex types other than float/double");
  }
}

// Simulation algorithm does not care about if matrix is unitary actually.
template <typename DType>
gate::Gate<DType> RandUnitaryGate(size_t circ_q_num, size_t gate_q_num) {
  std::vector<DType> data(1 << (1 << gate_q_num));
  for (int i = 0; i < data.size(); ++i) {
    data[i] = RandComplex<DType>();
  }
  if (gate_q_num == 1) {
    auto targ = RandNum<size_t>() % circ_q_num;
    return gate::Gate(targ, data.data());
  } else if (gate_q_num == 2) {
    auto targ0 = RandNum<size_t>() % circ_q_num,
         targ1 = RandNum<size_t>() % circ_q_num;
    return gate::Gate(targ0, targ1, data.data());
  } else {
    throw std::runtime_error("Not support qubit num >= 3");
  }
}

// @brief Run simulator with given gates. Meassure elapsed time.
// @returns Time used for simulation, in seconds.
template <typename DType>
inline double BenchSingleSimulator(
    sim::Simulator<DType> &simulator,
    const std::vector<gate::Gate<DType>> &gates) {
  using namespace std::chrono;
  auto start_time = steady_clock::now();
  for (const auto &gate : gates) {
    simulator.ApplyGate(gate);
  }
  auto end_time = steady_clock::now();
  duration<double> elapsed_time = end_time - start_time;
  return elapsed_time.count();
}
}  // namespace detail

// @brief Run different types of simulators with random generated gates.
//        Number of 1-bit and 2-bit gates are generated in fraction 2:1.
//        Print time information.
template <typename DType>
inline void BenchAllSimulators(size_t gate_num = 200, size_t repeat = 5) {
  using namespace sim;
  using namespace gate;
  using namespace detail;
  size_t test_q_nums[] = {5, 10, 15, 20, 25};
  for (size_t q_num : test_q_nums) {
    std::cout << q_num << " bits circuit:" << std::endl;

    std::vector<Gate<DType>> gates;
    for (size_t i = 0; i < gate_num / 3; ++i) {
      gates.push_back(std::move(RandUnitaryGate<DType>(q_num, 2)));
    }
    for (size_t i = gate_num / 3; i < gate_num; ++i) {
      gates.push_back(std::move(RandUnitaryGate<DType>(q_num, 1)));
    }
    std::shuffle(gates.begin(), gates.end(), gen);

    std::vector<Simulator<DType>> simulators;
    simulators.push_back(std::move(Simulator<DType>(q_num, BackendTag::NAIVE)));
    const auto &feat = Simulator<DType>::GetHardwareFeature();
    if (feat.HW_SSE && feat.HW_SSE2) {
      simulators.push_back(std::move(Simulator<DType>(q_num, BackendTag::SSE)));
    }

    for (auto &simulator : simulators) {
      // Skip naive simulator for large scale circuits.
      if (q_num >= 20 && simulator.GetBackendTag() == BackendTag::NAIVE) {
        std::cout << "\t" << simulator.GetName() << ":\tskipped" << std::endl;
        continue;
      }

      double duration = 0.0;
      for (size_t i = 0; i < repeat; ++i) {
        duration += BenchSingleSimulator(simulator, gates);
      }
      duration /= repeat;
      std::cout << "\t" << simulator.GetName() << ":\t" << std::fixed
                << std::setprecision(6) << duration << " s" << std::endl;
    }
    std::cout << std::endl;
  }
}
}  // namespace bench

#endif

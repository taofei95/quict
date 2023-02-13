#include "./bench_quict_sim.hpp"

using namespace bench;

int main() {
  bench::BenchAllSimulators<std::complex<float>>();
  bench::BenchAllSimulators<std::complex<double>>();
  return 0;
}

#include <gtest/gtest.h>

#include <iostream>
#include <complex>

#include "../simulator/simulator.hpp"

using namespace sim;

TEST(Misc, Specification) {
  Simulator<std::complex<float>> simulator(5, BackendTag::AUTO);
  std::cout << simulator.Spec() << std::endl;
}

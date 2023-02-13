#include "../../common/test_vec.hpp"

using namespace sim;

TEST(UnitaryGateNaive, Gate1BitComplexF32) {
  TestDType<std::complex<float>>(1, 5e-7, BackendTag::NAIVE);
}

TEST(UnitaryGateNaive, Gate1BitComplexF64) {
  TestDType<std::complex<double>>(1, 1e-7, BackendTag::NAIVE);
}

TEST(UnitaryGateNaive, Gate2BitComplexF32) {
  TestDType<std::complex<float>>(2, 5e-7, BackendTag::NAIVE);
}

TEST(UnitaryGateNaive, Gate2BitComplexF64) {
  TestDType<std::complex<double>>(2, 1e-7, BackendTag::NAIVE);
}

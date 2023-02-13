#include "../../../common/test_vec.hpp"

using namespace sim;

TEST(UnitaryGateSse, Gate1BitComplexF32) {
  TestDType<std::complex<float>>(1, 5e-7, BackendTag::SSE);
}

TEST(UnitaryGateSse, Gate1BitComplexF64) {
  TestDType<std::complex<double>>(1, 1e-7, BackendTag::SSE);
}

TEST(UnitaryGateSse, Gate2BitComplexF32) {
  TestDType<std::complex<float>>(2, 5e-7, BackendTag::SSE);
}

TEST(UnitaryGateSse, Gate2BitComplexF64) {
  TestDType<std::complex<double>>(2, 1e-7, BackendTag::SSE);
}

#include "../../../common/test_vec.hpp"

TEST(ApplyUnitaryGateNaive, ComplexF32) {
  TestDType<std::complex<float>>(5e-7, BackendTag::SSE);
}
TEST(ApplyUnitaryGateNaive, ComplexF64) {
  TestDType<std::complex<double>>(1e-7, BackendTag::SSE);
}

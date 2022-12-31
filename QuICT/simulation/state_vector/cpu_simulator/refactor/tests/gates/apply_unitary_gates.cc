#include <gtest/gtest.h>

#include <complex>

#include "../../simulator/simulator.hpp"
#include "../common/data_reader.hpp"

template <class T, class Str>
void TestPair(size_t q_num, Str desc_f_name, Str vec_f_name) {
  auto desc = test::ReadDesc<T>(desc_f_name);
  auto vec = test::ReadVec<T>(vec_f_name);

  simulator::Simulator<T> smltr(q_num);

  for (auto &gate : desc) {
    smltr.ApplyNormalizedGate(gate);
  }

  auto data = smltr.Data();
  for (int i = 0; i < vec.size(); ++i) {
    ASSERT_DOUBLE_EQ(std::abs(vec[i] - data[i]), 0.0);
  }
}

TEST(UniTaryGate, UnormalizedF32) {
  TestPair<std::complex<float>>(2, "desc_2bit-all_qubit2_size20",
                                "vec_2bit-all_qubit2_size20");
}

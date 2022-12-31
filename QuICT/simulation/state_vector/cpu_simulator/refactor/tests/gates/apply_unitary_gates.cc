#include <gtest/gtest.h>

#include <complex>
#include <iostream>

#include "../../simulator/simulator.hpp"
#include "../common/data_reader.hpp"

template <class T, class Str>
void TestPair(size_t q_num, Str desc_f_name, Str vec_f_name) {
  auto desc = test::ReadDesc<T>(desc_f_name);
  auto cmp_vec = test::ReadVec<T>(vec_f_name);
  ASSERT_TRUE(desc.size() > 0);
  ASSERT_TRUE(cmp_vec.size() > 0);

  simulator::Simulator<T> smltr(q_num);

  for (auto &gate : desc) {
    smltr.ApplyUnormalizedGate(gate);
  }

  auto res_data = smltr.Data();
  for (int i = 0; i < cmp_vec.size(); ++i) {
    ASSERT_NEAR(std::abs(cmp_vec[i] - res_data[i]), 0.0, 1e-7);
  }
}

TEST(UniTaryGate, UnormalizedF32) {
  TestPair<std::complex<float>>(2, "../tests/data/desc_2bit-all_qubit2_size20",
                                "../tests/data/vec_2bit-all_qubit2_size20");
}

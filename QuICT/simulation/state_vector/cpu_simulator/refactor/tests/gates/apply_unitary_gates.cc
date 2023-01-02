#include <gtest/gtest.h>

#include <complex>
#include <filesystem>
#include <iostream>
#include <regex>

#include "../../simulator/simulator.hpp"
#include "../common/data_reader.hpp"

namespace fs = std::filesystem;

template <class T, class Str>
void TestPair(size_t q_num, Str desc_f_name, Str vec_f_name, double eps) {
  auto desc = test::ReadDesc<T>(desc_f_name);
  auto cmp_vec = test::ReadVec<T>(vec_f_name);
  ASSERT_TRUE(desc.size() > 0) << "desc f name: " << desc_f_name << "\n";
  ASSERT_TRUE(cmp_vec.size() == (1ULL << q_num))
      << "vec f name: " << vec_f_name << "\n";

  sim::Simulator<T> simulator(q_num);

  for (auto &gate : desc) {
    simulator.ApplyGate(gate);
  }

  auto res_data = simulator.GetStateVector();
  for (int i = 0; i < cmp_vec.size(); ++i) {
    ASSERT_NEAR(std::abs(cmp_vec[i] - res_data[i]), 0.0, eps)
        << "Desc file name: " << desc_f_name << "\n";
  }
}

fs::path GetDataPath() {
  fs::path cwd = fs::current_path();
  fs::path data_dir = cwd.parent_path() / "tests" / "data";
  return data_dir;
}

size_t GetQubitNum(const std::string &f_name) {
  static std::regex rgx("(qubit)(\\w+)");
  std::smatch matches;
  assert(std::regex_search(f_name, matches, rgx));
  assert(matches.size() == 3);
  return std::stoi(matches[2].str());
}

template <class DType>
void TestDType(double eps) {
  auto data_dir = GetDataPath();
  for (auto &it : fs::directory_iterator(data_dir)) {
    auto desc_f_name = it.path().filename().string();
    if (desc_f_name.substr(0, 3) == "vec") {
      continue;
    }
    std::string vec_f_name = "vec_" + desc_f_name.substr(5);
    desc_f_name = (data_dir / desc_f_name).string();
    vec_f_name = (data_dir / vec_f_name).string();
    size_t q_num = GetQubitNum(desc_f_name);
    TestPair<DType>(q_num, desc_f_name, vec_f_name, eps);
  }
}

TEST(ApplyUnitaryGate, ComplexF32) { TestDType<std::complex<float>>(5e-7); }
TEST(ApplyUnitaryGate, ComplexF64) { TestDType<std::complex<double>>(1e-7); }

#ifndef QUICT_CPU_SIM_BACKEND_TESTS_COMMON_DATA_READRE_H
#define QUICT_CPU_SIM_BACKEND_TESTS_COMMON_DATA_READRE_H

#include <cassert>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../gate/gate.hpp"

namespace test {
template <class T, class Str>
std::vector<T> ReadVec(Str f_name) {
  std::fstream file(f_name);
  std::string line;
  std::vector<T> res;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    T val;
    if (line[0] == '(') {
      char lp;
      iss >> lp;
    }
    iss >> val;
    res.push_back(val);
  }
  return res;
}

template <class T, class Str>
std::vector<gate::Gate<T>> ReadDesc(Str f_name) {
  std::fstream file(f_name);
  std::string line;
  std::vector<gate::Gate<T>> res;
  std::string tmp1, tmp2, tag;
  int q_num;
  T val;
  char para;

  std::getline(file, line);
  std::istringstream iss(line);
  iss >> tmp1 >> q_num;

  while (std::getline(file, line)) {
    std::vector<T> gate_data;
    iss = std::istringstream(line);
    iss >> tmp1 >> tag >> tmp2;
    while (!std::isalpha(tag.back())) {
      tag.pop_back();
    }
    if (tag == "unitary") {
      for (int64_t i = 0; i < (2LL << q_num); ++i) {
        para = '\0';
        while (para != '(') {
          iss >> para;
        }
        iss >> val;
        gate_data.push_back(val);
      }
    } else {
      throw std::runtime_error("Gate type not supported yet");
    }
    res.emplace_back(gate::Gate<T>(q_num, gate_data.data()));
  }
  return res;
}
}  // namespace test

#endif

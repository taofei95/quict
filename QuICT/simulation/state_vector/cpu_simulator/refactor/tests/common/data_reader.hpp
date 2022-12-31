#ifndef QUICT_CPU_SIM_BACKEND_TESTS_COMMON_DATA_READRE_H
#define QUICT_CPU_SIM_BACKEND_TESTS_COMMON_DATA_READRE_H

#include <cassert>
#include <cctype>
#include <cstddef>
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

  std::getline(file, line);
  std::istringstream iss(line);
  iss >> tmp1 >> q_num;

  while (std::getline(file, line)) {
    std::vector<T> gate_data;
    iss = std::istringstream(line);
    iss >> tmp1 >> tag;  // `tag: xxx` part
    while (!std::isalpha(tag.back())) {
      tag.pop_back();
    }
    if (tag == "unitary") {
      int gq_num = 0;
      std::string targ_str = "";
      size_t targ[2];
      iss >> tmp1;  // `targ: ` part
      while (targ_str != ";") {
        iss >> targ_str;
        if (targ_str != ";") {
          targ[gq_num++] = std::stoi(targ_str);
        }
      }
      iss >> tmp1;  //`data: ` part
      for (int64_t i = 0; i < (1ULL << (q_num << 1)); ++i) {
        iss >> val;
        gate_data.push_back(val);
      }
      if (gq_num == 1) {
        res.emplace_back(targ[0], gate_data.data());
      } else if (gq_num == 2) {
        res.emplace_back(targ[0], targ[1], gate_data.data());
      } else {
        throw std::runtime_error("Not support qubit >= 3!");
      }
    } else {
      throw std::runtime_error("Gate type not supported yet!");
    }
  }
  return res;
}
}  // namespace test

#endif

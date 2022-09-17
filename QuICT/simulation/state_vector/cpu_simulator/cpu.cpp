//
// Created by Ci Lei on 2021-10-27.
//

#ifndef SIM_BACK_PORT_H
#define SIM_BACK_PORT_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>
#include <cstdint>
#include <tuple>
#include <vector>

#include "./src/circuit_simulator.h"
#include "./src/utility.h"

namespace py = pybind11;

namespace QuICT {
/*
 * CircuitSimulator wrapper which automatically wraps return value of
 * `run` method of CircuitSimulator as a numpy array.
 * */
class CircuitSimulatorBind : public CircuitSimulator<double> {
 public:
  CircuitSimulatorBind() : CircuitSimulator<double>() {}

  inline std::tuple<py::array_t<std::complex<double>>, std::vector<int>>
  run_numpy(uint64_t qubit_num,
            const std::vector<GateDescription<double>> &gate_desc_vec,
            bool keep_state) {
    std::vector<int> measure_res;
    std::complex<double> *raw_ptr =
        run(qubit_num, gate_desc_vec, measure_res, keep_state);

    py::capsule auto_delete_wrapper(raw_ptr, [](void *ptr) {
      auto data_ptr = reinterpret_cast<std::complex<double> *>(ptr);
      delete[] data_ptr;
    });

    auto amplitude = py::array_t<std::complex<double>>(
        {1ULL << this->qubit_num_},      // shape
        {sizeof(std::complex<double>)},  // stride
        raw_ptr,                         // data pointer
        auto_delete_wrapper              // numpy array reference this parent
    );
    return std::make_tuple(amplitude, measure_res);
  }
};
}  // namespace QuICT

PYBIND11_MODULE(sim_back_bind, m) {
  using sim_class = QuICT::CircuitSimulatorBind;
  using desc_class = QuICT::GateDescription<double>;
  // Export GateDescription interface.
  py::class_<desc_class>(m, "GateDescription")
      .def(py::init<const char *, std::vector<uint64_t>,
                    std::vector<std::complex<double>>>())
      .def_readwrite("gateName", &desc_class::gate_name_)
      .def_readwrite("affectArgs", &desc_class::affect_args_)
      .def_readwrite("dataPtr", &desc_class::data_ptr_);
  // Export CircuitSimulator interface using wrapper class.
  py::class_<sim_class>(m, "CircuitSimulator")
      .def(py::init<>())
      .def("name", &sim_class::name)
      .def("run", &sim_class::run_numpy)
      .def("sample", &sim_class::sample);
}

#endif  // SIM_BACK_PORT_H

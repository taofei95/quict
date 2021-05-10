#ifndef GRAPH_STRUCTURE_GRAPH_PYTHON_PORT_H
#define GRAPH_STRUCTURE_GRAPH_PYTHON_PORT_H

#include "graph.h"
#include "../extern/pybind11/include/pybind11/pybind11.h"
#include "../extern/pybind11/include/pybind11/stl.h"
#include "../extern/pybind11/include/pybind11/stl_bind.h"
#include <Python.h>
#include <string>
#include <memory>

namespace py = pybind11;

template<typename vertex_label_t, typename edge_data_t>
void expose_edge(py::module &m, const std::string &type_str_suffix) {
    using clazz = graph::edge_t<vertex_label_t, edge_data_t>;
    std::string type_str = "edge" + type_str_suffix;
    py::class_<clazz, std::shared_ptr<clazz>>(m, type_str.c_str())
            .def_readwrite("from", &clazz::from_)
            .def_readwrite("to", &clazz::to_)
            .def_readwrite("data", &clazz::data_)
            .def("print_info", &clazz::print_info);
}

template<typename vertex_label_t, typename edge_data_t>
void expose_graph(py::module &m, const std::string &type_str_suffix) {
    using clazz = graph::directed_graph<vertex_label_t, edge_data_t>;
    std::string type_str = "directed_graph" + type_str_suffix;

    expose_edge<vertex_label_t, edge_data_t>(m, type_str);

    py::class_<clazz>(m, type_str.c_str())
            .def(py::init<>())
            .def("print_info", &clazz::print_info)
            .def("add_edge", &clazz::add_edge)
            .def("add_vertex", &clazz::add_vertex)
            .def("edge_cnt", &clazz::edge_cnt)
            .def("edges_from", &clazz::edges_from)
            .def("out_deg_of", &clazz::oud_deg_of)
            .def("edges_to", &clazz::edges_to)
            .def("in_deg_of", &clazz::in_deg_of)
            .def("edges", &clazz::edges)
            .def("dijkstra", &clazz::dijkstra);
}

PYBIND11_MODULE(graph_structure, m) {
    expose_graph<py::int_, py::int_>(m, "_vertex_label_int");
    expose_graph<py::str, py::int_>(m, "_vertex_label_str");
}


#endif //GRAPH_STRUCTURE_GRAPH_PYTHON_PORT_H

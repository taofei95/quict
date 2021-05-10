cmake_minimum_required(VERSION 3.16)
project(graph_structure)

set(CMAKE_CXX_STANDARD 17)

find_package(Python3 COMPONENTS Interpreter Development)

add_subdirectory(../extern/pybind11 ../extern/pybind11 EXCLUDE_FROM_ALL)
include_directories(${Python3_INCLUDE_DIRS})

#add_library(graph_structure SHARED graph.h graph_python_port.cpp graph_python_port.h)
#add_executable(graph_structure_test graph.h test_run.cpp)


pybind11_add_module(graph_structure graph_python_port.h graph_python_port.cpp ${Python3_INCLUDE_DIRS})
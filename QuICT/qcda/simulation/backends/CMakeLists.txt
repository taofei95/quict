cmake_minimum_required(VERSION 3.16)
project(sim_back_bind CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize -fopenmp -mavx -mavx2 -mfma")

find_package(Python3 COMPONENTS Interpreter Development)
add_subdirectory(../../../utility/extern/pybind11 ../cmake-build-pybind11/sim_back_bind EXCLUDE_FROM_ALL)
include_directories(${Python3_INCLUDE_DIRS})

pybind11_add_module(sim_back_bind port.cpp src/circuit_simulator.h src/q_state_simulator.h src/matricks_simulator.h
        src/tiny_simulator.h src/q_state.h src/q_state_set.h src/gate.h src/utility.h)

target_compile_definitions(sim_back_bind PRIVATE VERSION_INFO=${BUILD_VERSION_INFO})
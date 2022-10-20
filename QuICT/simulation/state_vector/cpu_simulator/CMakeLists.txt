cmake_minimum_required(VERSION 3.16)
project(sim_back_bind CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize -fopenmp -mavx -mavx2 -mfma")

find_package(pybind11 REQUIRED)

pybind11_add_module(sim_back_bind cpu.cpp src/circuit_simulator.h src/q_state_simulator.h src/matricks_simulator.h
        src/tiny_simulator.h src/q_state.h src/q_state_set.h src/gate.h src/utility.h)
target_compile_definitions(sim_back_bind PRIVATE VERSION_INFO=${BUILD_VERSION_INFO})
target_link_libraries(sim_back_bind PRIVATE pybind11::module)

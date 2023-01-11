cmake_minimum_required(VERSION 3.0)
project(sim_back_bind CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize -mavx -mavx2 -mfma")

find_package(pybind11 REQUIRED)

pybind11_add_module(sim_back_bind cpu.cpp src/circuit_simulator.h src/q_state_simulator.h src/matricks_simulator.h
        src/tiny_simulator.h src/q_state.h src/q_state_set.h src/gate.h src/utility.h)
target_compile_definitions(sim_back_bind PRIVATE VERSION_INFO=${BUILD_VERSION_INFO})
target_link_libraries(sim_back_bind PRIVATE pybind11::module OpenMP::OpenMP_CXX)

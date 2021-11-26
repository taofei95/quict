cmake_minimum_required(VERSION 3.16)
project(sim_back CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_testing()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize -fopenmp -mavx -mavx2 -mfma")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize -mavx -mavx2 -mfma")


add_subdirectory(../../../../../QuICT/utility/extern/googletest ./cmake-build-gtest)
#include_directories(${gtest_SOURCE_DIR}/include)
include(GoogleTest)

add_executable(misc_test miscellaneous_test.cpp gate.h utility.h matricks_simulator.h)
target_link_libraries(misc_test gtest_main)

add_executable(matricks_sim_test matricks_sim_test.cpp matricks_simulator.h gate.h utility.h test_utility.h)
target_link_libraries(matricks_sim_test gtest_main)

add_executable(matricks_sim_perf matricks_sim_perf.cpp matricks_simulator.h gate.h utility.h)
target_link_libraries(matricks_sim_perf gtest_main)

add_executable(gate_test gate_test.cpp matricks_simulator.h gate.h utility.h)
target_link_libraries(gate_test gtest_main)

add_executable(index_test index_test.cpp gate.h utility.h)
target_link_libraries(index_test gtest_main)

add_executable(tiny_sim_test tiny_sim_test.cpp tiny_simulator.h gate.h utility.h test_utility.h)
target_link_libraries(tiny_sim_test gtest_main)

add_executable(q_state_test q_state_test.cpp gate.h utility.h test_utility.h q_state.h q_state_set.h)
target_link_libraries(q_state_test gtest_main)

add_executable(circuit_sim_test circuit_sim_test.cpp circuit_simulator.h q_state.h q_state_set.h
        q_state_simulator.h gate.h utility.h test_utility.h)
target_link_libraries(circuit_sim_test gtest_main)
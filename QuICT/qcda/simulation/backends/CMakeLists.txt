cmake_minimum_required(VERSION 3.16)
project(sim_back CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize -fopenmp -mavx -mavx2 -mfma")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize -mavx -mavx2 -mfma")


add_subdirectory(../../../../QuICT/utility/extern/googletest ./cmake-build-gtest)
include_directories(${gtest_SOURCE_DIR}/include)

add_executable(misc_test miscellaneous_test.cpp gate.h utility.h
        ../../../../QuICT/utility/extern/googletest/googletest/src/gtest_main.cc matricks_simulator.h)
add_dependencies(misc_test gtest)
target_link_libraries(misc_test gtest)

add_executable(matricks_sim_test matricks_sim_test.cpp matricks_simulator.h gate.h utility.h
        ../../../../QuICT/utility/extern/googletest/googletest/src/gtest_main.cc)
add_dependencies(matricks_sim_test gtest)
target_link_libraries(matricks_sim_test gtest)

add_executable(matricks_sim_perf matricks_sim_perf.cpp matricks_simulator.h gate.h utility.h
        ../../../../QuICT/utility/extern/googletest/googletest/src/gtest_main.cc)
add_dependencies(matricks_sim_perf gtest)
target_link_libraries(matricks_sim_perf gtest)

add_executable(gate_test gate_test.cpp matricks_simulator.h gate.h utility.h
        ../../../../QuICT/utility/extern/googletest/googletest/src/gtest_main.cc)
add_dependencies(gate_test gtest)
target_link_libraries(gate_test gtest)

add_executable(index_test index_test.cpp gate.h utility.h
        ../../../../QuICT/utility/extern/googletest/googletest/src/gtest_main.cc)
add_dependencies(index_test gtest)
target_link_libraries(index_test gtest)
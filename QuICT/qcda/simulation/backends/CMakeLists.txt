cmake_minimum_required(VERSION 3.16)
project(sim_back CXX)

set(CMAKE_CXX_STANDARD 17)

enable_testing()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(../../../../QuICT/utility/extern/googletest ./cmake-build-gtest )

include_directories(../../../../QuICT/utility/extern/googletest/googletest/include)
include_directories(../../../../QuICT/utility/extern/googletest/googlemock/include)

add_library(sim_back SHARED gate.h simulator.h utility.h)

add_executable(sim_back_test simulation_backends_test.cpp gate.h simulator.h utility.h
        ../../../../QuICT/utility/extern/googletest/googletest/src/gtest_main.cc)
target_link_libraries(sim_back_test gtest_main)

add_executable(sim_back_perf simulation_backends_perf.cpp gate.h simulator.h utility.h
        ../../../../QuICT/utility/extern/googletest/googletest/src/gtest_main.cc)
target_link_libraries(sim_back_perf gtest_main)

set_target_properties(sim_back PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(sim_back_test PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(sim_back_perf PROPERTIES LINKER_LANGUAGE CXX)
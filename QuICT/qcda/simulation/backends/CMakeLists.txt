cmake_minimum_required(VERSION 3.16)
project(simulation_backends CXX)

set(CMAKE_CXX_STANDARD 17)

enable_testing()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(../../../../QuICT/utility/extern/googletest ./cmake-build-gtest )

include_directories(../../../../QuICT/utility/extern/googletest/googletest/include)
include_directories(../../../../QuICT/utility/extern/googletest/googlemock/include)

add_library(simulation_backends SHARED gate.h simulator.h utility.h)

add_executable(simulation_backends_test ../../../../QuICT/utility/extern/googletest/googletest/src/gtest_main.cc
        simulation_backends_test.cpp gate.h simulator.h utility.h)
target_link_libraries(simulation_backends_test gtest_main)

#include(GoogleTest)
#gtest_discover_tests(simulation_backends_test)
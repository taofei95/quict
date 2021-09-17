cmake_minimum_required(VERSION 3.16)
project(sim_back CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)


#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize -fopenmp -mavx -mavx2 -mfma")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize -mavx -mavx2 -mfma")



add_subdirectory(../../../../QuICT/utility/extern/googletest ./cmake-build-gtest)
include_directories(${gtest_SOURCE_DIR}/include)

add_executable(mono_sim_test mono_sim_test.cpp monotonous_simulator.h gate.h utility.h
        ../../../../QuICT/utility/extern/googletest/googletest/src/gtest_main.cc)
add_dependencies(mono_sim_test gtest)
target_link_libraries(mono_sim_test gtest_main)

add_executable(mono_sim_perf mono_sim_perf.cpp monotonous_simulator.h gate.h utility.h
        ../../../../QuICT/utility/extern/googletest/googletest/src/gtest_main.cc)
add_dependencies(mono_sim_perf gtest)
target_link_libraries(mono_sim_perf gtest)

add_executable(misc_test miscellaneous_test.cpp monotonous_simulator.h gate.h utility.h
        ../../../../QuICT/utility/extern/googletest/googletest/src/gtest_main.cc hybrid_simulator.h)
add_dependencies(misc_test gtest)
target_link_libraries(misc_test gtest)

add_executable(hybrid_sim_test hybrid_sim_test.cpp hybrid_simulator.h gate.h utility.h
        ../../../../QuICT/utility/extern/googletest/googletest/src/gtest_main.cc)
add_dependencies(hybrid_sim_test gtest)
target_link_libraries(hybrid_sim_test gtest)

add_executable(hybrid_sim_perf hybrid_sim_perf.cpp hybrid_simulator.h gate.h utility.h
        ../../../../QuICT/utility/extern/googletest/googletest/src/gtest_main.cc)
add_dependencies(hybrid_sim_perf gtest)
target_link_libraries(hybrid_sim_perf gtest)

add_executable(gate_test gate_test.cpp hybrid_simulator.h gate.h utility.h
        ../../../../QuICT/utility/extern/googletest/googletest/src/gtest_main.cc)
add_dependencies(gate_test gtest)
target_link_libraries(gate_test gtest)

add_executable(index_test index_test.cpp gate.h utility.h
        ../../../../QuICT/utility/extern/googletest/googletest/src/gtest_main.cc)
add_dependencies(index_test gtest)
target_link_libraries(index_test gtest)
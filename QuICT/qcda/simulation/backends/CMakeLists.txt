cmake_minimum_required(VERSION 3.16)
project(sim_back CXX)

set(CMAKE_CXX_STANDARD 17)

enable_testing()

#include(ProcessorCount)
#ProcessorCount(NPROC)
#if (NOT NPROC EQUAL 0)
#    MATH(EXPR NPROC_L "${NPROC}-4")
#    if (${NPROC_L} LESS 1)
#        set(NPROC_L 1)
#    endif ()
#    message("Set omp threads = ${NPROC_L}")
#    add_compile_definitions(OMP_NPROC=${NPROC_L})
#endif ()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qpar /arch:AVX /arch:AVX2")
else ()
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize -fopenmp -mavx -mavx2 -mfma")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize -mavx -mavx2 -mfma")
endif ()


add_subdirectory(../../../../QuICT/utility/extern/googletest ./cmake-build-gtest)

include_directories(../../../../QuICT/utility/extern/googletest/googletest/include)
include_directories(../../../../QuICT/utility/extern/googletest/googlemock/include)

#add_library(mono_sim SHARED gate.h monotune_simulator.h utility.h hybrid_simulator.h)

add_executable(mono_sim_test mono_sim_test.cpp monotonous_simulator.h gate.h utility.h
        ../../../../QuICT/utility/extern/googletest/googletest/src/gtest_main.cc)
target_link_libraries(mono_sim_test gtest_main)

add_executable(mono_sim_perf mono_sim_perf.cpp monotonous_simulator.h gate.h utility.h
        ../../../../QuICT/utility/extern/googletest/googletest/src/gtest_main.cc)
target_link_libraries(mono_sim_perf gtest_main)

add_executable(misc_test miscellaneous_test.cpp monotonous_simulator.h gate.h utility.h
        ../../../../QuICT/utility/extern/googletest/googletest/src/gtest_main.cc hybrid_simulator.h)
target_link_libraries(misc_test gtest_main)

add_executable(hybrid_sim_test hybrid_sim_test.cpp hybrid_simulator.h gate.h utility.h
        ../../../../QuICT/utility/extern/googletest/googletest/src/gtest_main.cc)
target_link_libraries(hybrid_sim_test gtest_main)

add_executable(gate_test gate_test.cpp hybrid_simulator.h gate.h utility.h
        ../../../../QuICT/utility/extern/googletest/googletest/src/gtest_main.cc)
target_link_libraries(gate_test gtest_main)

#set_target_properties(mono_sim PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(mono_sim_test PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(mono_sim_perf PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(misc_test PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(hybrid_sim_test PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(gate_test PROPERTIES LINKER_LANGUAGE CXX)
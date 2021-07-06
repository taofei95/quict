cmake_minimum_required(VERSION 3.16)
project(sim_back CXX)

set(CMAKE_CXX_STANDARD 17)

enable_testing()

include(ProcessorCount)
ProcessorCount(NPROC)
if (NOT NPROC EQUAL 0)
    MATH(EXPR NPROC_L "${NPROC}-4")
    if (${NPROC_L} LESS 1)
        set(NPROC_L 1)
    endif ()
    message("Set omp threads = ${NPROC_L}")
    add_compile_definitions(OMP_NPROC=${NPROC_L})
endif ()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qpar /arch:AVX /arch:AVX2")
else ()
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize -fopenmp -mavx -mavx2 -mfma")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -save-temps -ftree-vectorize -mavx -mavx2 -mfma")
endif ()


add_subdirectory(../../../../QuICT/utility/extern/googletest ./cmake-build-gtest)

include_directories(../../../../QuICT/utility/extern/googletest/googletest/include)
include_directories(../../../../QuICT/utility/extern/googletest/googlemock/include)

add_library(sim_back SHARED gate.h monotune_simulator.h utility.h hybrid_simulator.h)

add_executable(sim_back_test simulation_backends_test.cpp gate.h monotune_simulator.h utility.h
        ../../../../QuICT/utility/extern/googletest/googletest/src/gtest_main.cc hybrid_simulator.h)
target_link_libraries(sim_back_test gtest_main)

add_executable(sim_back_perf simulation_backends_perf.cpp gate.h monotune_simulator.h utility.h
        ../../../../QuICT/utility/extern/googletest/googletest/src/gtest_main.cc hybrid_simulator.h)
target_link_libraries(sim_back_perf gtest_main)

add_executable(misc_test miscllaneous_test.cpp  gate.h monotune_simulator.h utility.h
        ../../../../QuICT/utility/extern/googletest/googletest/src/gtest_main.cc hybrid_simulator.h)
target_link_libraries(misc_test gtest_main)

set_target_properties(sim_back PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(sim_back_test PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(sim_back_perf PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(misc_test PROPERTIES LINKER_LANGUAGE CXX)
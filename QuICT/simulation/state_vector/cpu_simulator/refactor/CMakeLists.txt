cmake_policy(SET CMP0048 NEW)
project(quict_cpu_sim_backend VERSION 1.0.0 LANGUAGES CXX)

include(CheckCXXCompilerFlag)

cmake_minimum_required(VERSION 3.0)

# Only support 64bit mode.
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message("Please switch to x64 build.")
  return()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Globally enable OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(SIMULATOR_SRC 
  gate/gate.hpp 
  gate/gate_attr.hpp
  simulator/simulator.hpp 
  simulator/apply_gate/delegate.hpp
  simulator/apply_gate/impl/naive.hpp
  simulator/apply_gate/impl/x86_64/sse.hpp
  )

# Detect CPU features
if(MSVC)
  # SSE
  set(QUICT_SSE_FLAG "/arch:SSE")
  set(QUICT_SSE2_FLAG "/arch:SSE2")

  # AVX
  set(QUICT_AVX_FLAG "/arch:AVX")
  set(QUICT_AVX2_FLAG "/arch:AVX2")

  # AVX512F
  set(QUICT_AVX512F_FLAG "/arch:AVX512")
else()
  # SSE
  set(QUICT_SSE_FLAG "-msse")
  set(QUICT_SSE2_FLAG "-msse2")

  # AVX
  set(QUICT_AVX_FLAG "-mavx")
  set(QUICT_AVX2_FLAG "-mavx2")

  # AVX512F
  set(QUICT_AVX512F_FLAG "-mavx512f")
endif()

macro(quict_check_flag flag_name)
  set(flag "${QUICT_${flag_name}_FLAG}")
  check_cxx_compiler_flag("${flag}" "QUICT_SUPPORT_${flag_name}")

  if(${QUICT_SUPPORT_${flag_name}})
    message(STATUS "QuICT ${flag_name} flags: ${QUICT_${flag_name}_FLAG}")
  endif()
  unset(flag)
endmacro()

quict_check_flag("SSE")
quict_check_flag("SSE2")
quict_check_flag("AVX")
quict_check_flag("AVX2")
quict_check_flag("AVX512F")

macro(quict_target_enable name)
  set(_TMP_FEAT_FLAG_LIST "")
  foreach(feat IN ITEMS ${ARGN})
    if(${QUICT_SUPPORT_${feat}})
      list(APPEND _TMP_FEAT_FLAG_LIST "${QUICT_${feat}_FLAG}")
    else()
      message(STATUS "Host does not support ${feat}. Compile ${name} without it.")
    endif()
  endforeach()
  message(STATUS "Target ${name} compiler feature flags: ${_TMP_FEAT_FLAG_LIST}")
  foreach(flag IN ITEMS ${_TMP_FEAT_FLAG_LIST})
    target_compile_options(${name} PUBLIC "${flag}")
    target_link_options(${name} PUBLIC "${flag}")
  endforeach()
  unset(_TMP_FEAT_FLAG_LIST)
endmacro()

# Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

if(NOT BUILD_TESTS)
  message(STATUS "Skip all tests.")
else()
  message(STATUS "Build all tests.")

  enable_testing()
  include(GoogleTest)

  macro(quict_add_test name)
    add_executable("${name}" "${ARGN}")
    target_link_libraries("${name}" GTest::gtest_main)
    
    gtest_discover_tests("${name}")
  endmacro()

  quict_add_test(spec ${SIMULATOR_SRC} tests/spec.cc)
  quict_target_enable(spec "SSE" "SSE2" "AVX" "AVX2")

  quict_add_test(unitary_gate_naive ${SIMULATOR_SRC} tests/gates/unitary/naive.cc)

  if(QUICT_SUPPORT_SSE AND QUICT_SUPPORT_SSE2)
    quict_add_test(unitary_gate_sse ${SIMULATOR_SRC} tests/gates/unitary/x86_64/sse.cc)
    quict_target_enable(unitary_gate_sse "SSE" "SSE2")
  endif()
endif()

add_executable(bench_quict_sim ${SIMULATOR_SRC} bench/bench_quict_sim.hpp bench/bench_quict_sim.cc)
quict_target_enable(bench_quict_sim "SSE" "SSE2" "AVX" "AVX2")


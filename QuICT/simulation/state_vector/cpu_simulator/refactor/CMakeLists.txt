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

macro(quict_check_flag flag flag_name)
  check_cxx_compiler_flag(${flag} "QUICT_SUPPORT_${flag_name}")
  message(STATUS "QuICT ${flag_name} flags: ${QUICT_${flag_name}_FLAG}")
endmacro()

quict_check_flag("${QUICT_SSE_FLAG}" "SSE")
quict_check_flag("${QUICT_SSE2_FLAG}" "SSE2")
quict_check_flag("${QUICT_AVX_FLAG}" "AVX")
quict_check_flag("${QUICT_AVX2_FLAG}" "AVX2")
quict_check_flag("${QUICT_AVX512F_FLAG}" "AVX512F")

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
  target_compile_options(spec PRIVATE ${QUICT_SSE_FLAG} ${QUICT_SSE2_FLAG} ${QUICT_AVX_FLAG} ${QUICT_AVX2_FLAG} ${QUICT_AVX512F_FLAG})

  quict_add_test(unitary_gate_naive ${SIMULATOR_SRC} tests/gates/unitary/naive.cc)

  if(QUICT_SUPPORT_SSE AND QUICT_SUPPORT_SSE2)
    quict_add_test(unitary_gate_sse ${SIMULATOR_SRC} tests/gates/unitary/x86_64/sse.cc)
    target_compile_options(unitary_gate_sse PRIVATE ${QUICT_SSE_FLAG} ${QUICT_SSE2_FLAG})
  endif()
endif()


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
  set(SSE_FLAG "/arch:SSE")
  set(SSE2_FLAG "/arch:SSE2")
else()
  # SSE
  set(SSE_FLAG "-msse")
  set(SSE2_FLAG "-msse2")
endif()

set(QUICT_SSE_FAMILY_FLAGS "${SSE_FLAG} ${SSE2_FLAG}")

macro(quict_check_flag flag flag_name)
  check_cxx_compiler_flag(${flag} "QUICT_SUPPORT_${flag_name}")
  if("${QUICT_SUPPORT_${flag_name}}")
    add_compile_definitions(QUICT_SUPPORT_${flag_name}=1)
  else()
    add_compile_definitions(QUICT_SUPPORT_${flag_name}=0)
  endif()
endmacro()

quict_check_flag("${SSE_FLAG}" "SSE")
quict_check_flag("${SSE2_FLAG}" "SSE2")

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

  quict_add_test(unitary_gate_naive ${SIMULATOR_SRC} tests/gates/unitary/naive.cc)

  if(QUICT_SUPPORT_SSE_FAMILY)
    quict_add_test(unitary_gate_sse ${SIMULATOR_SRC} tests/gates/unitary/x86_64/sse.cc)
  endif()
endif()


project(quict_cpu_sim_backend LANGUAGES C CXX)

cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(SIMULATOR_SRC simulator/simulator.hpp gate/gate.hpp gate/gate_attr.hpp)

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

enable_testing()

set(U_GATE_TEST_SRC ${SIMULATOR_SRC} tests/gates/apply_unitary_gates.cc)
add_executable(apply_unitary_gates ${U_GATE_TEST_SRC})
target_link_libraries(apply_unitary_gates GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(apply_unitary_gates)



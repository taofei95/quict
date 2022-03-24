cmake_minimum_required(VERSION 3.16)
project(initial_state_preparation)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(TBB REQUIRED COMPONENTS tbb)

add_library(initial_state_preparation SHARED initial_state_preparation.cpp ${TBB_INCLUDE_DIRS})


set_target_properties(initial_state_preparation PROPERTIES PREFIX "")
set_target_properties(initial_state_preparation PROPERTIES OUTPUT_NAME "initial_state_preparation_cdll")

target_link_libraries(initial_state_preparation TBB::tbb)

target_compile_definitions(initial_state_preparation PRIVATE VERSION_INFO=${BUILD_VERSION_INFO})
cmake_minimum_required(VERSION 3.16)
project(backends)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(TBB REQUIRED COMPONENTS tbb)

add_library(backends SHARED dll.cpp ${TBB_INCLUDE_DIRS})


set_target_properties(backends PROPERTIES PREFIX "")
set_target_properties(backends PROPERTIES OUTPUT_NAME "quick_operator_cdll")

target_link_libraries(backends TBB::tbb)

target_compile_definitions(backends PRIVATE VERSION_INFO=${BUILD_VERSION_INFO})
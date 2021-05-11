cmake_minimum_required(VERSION 3.16)
project(graph_structure)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development)

add_subdirectory(../extern/pybind11 ../extern/pybind11 EXCLUDE_FROM_ALL)
include_directories(${Python3_INCLUDE_DIRS})


pybind11_add_module(graph_structure graph_python_port.h graph_python_port.cpp ${Python3_INCLUDE_DIRS})

# remove compiler-specific suffix
# if (CMAKE_HOST_APPLE)
# set_target_properties(graph_structure PROPERTIES SUFFIX ".dylib")
# elseif(CMAKE_HOST_WIN32)
# set_target_properties(graph_structure PROPERTIES SUFFIX ".dll")
# else()
# set_target_properties(graph_structure PROPERTIES SUFFIX ".so")
# endif()

# set_target_properties(graph_structure PROPERTIES SUFFIX ".so")

target_compile_definitions(graph_structure PRIVATE VERSION_INFO=${BUILD_VERSION_INFO})
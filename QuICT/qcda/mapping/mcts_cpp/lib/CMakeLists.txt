cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mcts_cpp)
include_directories("${PROJECT_SOURCE_DIR}/include/")
aux_source_directory("${PROJECT_SOURCE_DIR}/src/" MCTS_SRC)
find_package(Torch REQUIRED)
find_package(OpenMP)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")


add_library(mcts  SHARED ${MCTS_SRC})
target_link_libraries(mcts "${TORCH_LIBRARIES}")
set_property(TARGET mcts PROPERTY CXX_STANDARD 14)
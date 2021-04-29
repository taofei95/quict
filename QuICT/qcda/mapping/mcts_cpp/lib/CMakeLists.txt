cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mcts_cpp)

option(TEST_MCTS "" on)
option(TEST_RL_MCTS "" off)
option(TEST_DATA_GENERATOR "" off)

include_directories("${PROJECT_SOURCE_DIR}/include/")
aux_source_directory("${PROJECT_SOURCE_DIR}/src/" MCTS_SRC)
find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)
if(Torch_FOUND AND OpenMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_library(mcts  SHARED ${MCTS_SRC})
target_link_libraries(mcts "${TORCH_LIBRARIES}")
set_property(TARGET mcts PROPERTY CXX_STANDARD 14)



if(TEST_MCTS)
    add_executable(test_mcts test_mcts_tree.cc ${MCTS_SRC})
    target_link_libraries(test_mcts "${TORCH_LIBRARIES}")
    set_property(TARGET test_mcts PROPERTY CXX_STANDARD 14)
endif()

if(TEST_RL_MCTS)
    add_executable(test_rl_mcts  test_rl_mcts_tree.cc ${MCTS_SRC})
    target_link_libraries(test_rl_mcts "${TORCH_LIBRARIES}")
    set_property(TARGET test_rl_mcts PROPERTY CXX_STANDARD 14)
endif()

if(TEST_DATA_GENERATOR)
    add_executable(test_data_generator  test_data_generator.cc ${MCTS_SRC})
    target_link_libraries(test_data_generator "${TORCH_LIBRARIES}")
    set_property(TARGET test_data_generator PROPERTY CXX_STANDARD 14)
endif()

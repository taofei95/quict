project(quict_cpu_sim_backend LANGUAGES C CXX)

cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_executable(tmp tmp_main.cc)


cmake_minimum_required(VERSION 3.16)
project(consumer_test LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(apg_sgm 1.0.0 REQUIRED)

add_executable(consumer_cpp main.cpp)
target_link_libraries(consumer_cpp PRIVATE apg_sgm::apg_sgm)

add_executable(consumer_c main.c)
target_link_libraries(consumer_c PRIVATE apg_sgm::apg_sgm)

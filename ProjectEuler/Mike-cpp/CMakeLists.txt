cmake_minimum_required(VERSION 3.16)

project(euler)

find_package(benchmark REQUIRED)

add_executable(euler 40.cc)

target_link_libraries(euler benchmark::benchmark)

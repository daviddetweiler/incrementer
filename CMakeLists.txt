cmake_minimum_required(VERSION 3.10)

project(incrementer)

find_package(Threads REQUIRED)

add_executable(test main.cpp)
target_link_libraries(test Threads::Threads)
set_property(TARGET test PROPERTY CXX_STANDARD 17)

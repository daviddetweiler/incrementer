cmake_minimum_required(VERSION 3.10)

project(incrementer C CXX ASM)

find_package(Threads REQUIRED)

add_compile_options(
    -Wall
    -mprefetchwt1
    -march=native
    -fcf-protection=none
)

set(CMAKE_BUILD_TYPE RelWithDebInfo)
add_executable(test main.cpp zipf_distribution.cpp lock_test.s spin_lock.s)
target_link_libraries(test Threads::Threads)
set_property(TARGET test PROPERTY CXX_STANDARD 17)

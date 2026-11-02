cmake_minimum_required(VERSION 4.4)
project(brainfah)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_executable(brainfah main.cpp)
target_compile_options(brainfah PRIVATE -Wall -Wextra -Werror -Wshadow -Wconversion)
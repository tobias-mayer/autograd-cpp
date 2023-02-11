cmake_minimum_required(VERSION 3.13)

project(autograd)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(dependencies)

add_library(autograd INTERFACE)
target_include_directories(autograd INTERFACE src)

add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(python_bindings)

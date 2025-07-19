cmake_minimum_required(VERSION 3.14)
project(sdr_pybind)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# pybind11
find_package(pybind11 REQUIRED)

# Add bindings
add_subdirectory(pybind11_bindings)

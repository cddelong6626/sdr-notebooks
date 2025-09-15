
# Build C++ libraries for usage in python notebooks and C++ SDR software
#  * sdrlib: C++ library including implementations of SDR algorithms intended for real-time processing
#  * py_sdrlib: Python library with bindings to sdrlib

cmake_minimum_required(VERSION 3.15)
project(sdr-notebooks LANGUAGES CXX)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(sdrlib)
add_subdirectory(py_bindings)
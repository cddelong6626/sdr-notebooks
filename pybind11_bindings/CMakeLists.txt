file(GLOB SRC_FILES dsp/*.cpp python/*.cpp)

pybind11_add_module(sdr_cpp ${SRC_FILES})

target_include_directories(sdr_cpp PRIVATE dsp)


cmake_minimum_required(VERSION 3.15)
project(dsp_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Add subdirectories
add_subdirectory(dsp)
add_subdirectory(python)
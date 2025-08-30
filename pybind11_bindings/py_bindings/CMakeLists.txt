# Build library using pybind11
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)
pybind11_add_module(py_sdrlib bindings.cpp)

# Allow usage of NumPy bindings
target_compile_definitions(py_sdrlib PRIVATE PYBIND11_NUMPY)

# Include KFR library
find_package(KFR CONFIG REQUIRED)
target_link_libraries(py_sdrlib PRIVATE kfr kfr_dsp)

# Link sdrlib C++ library
target_link_libraries(py_sdrlib PRIVATE sdrlib)

# Install python library into the python env's site packages
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
install(TARGETS py_sdrlib LIBRARY DESTINATION ${Python3_SITEARCH})

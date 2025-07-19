find_package(pybind11 REQUIRED)

pybind11_add_module(dsp_cpp bindings.cpp)

target_link_libraries(dsp_cpp PRIVATE dsp)

target_compile_definitions(dsp_cpp PRIVATE PYBIND11_NUMPY)
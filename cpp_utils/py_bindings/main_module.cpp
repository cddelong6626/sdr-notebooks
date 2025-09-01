#include <pybind11/pybind11.h>
#include <kfr/base.hpp>

namespace py = pybind11;

// Declarations of binding functions. These are defined in dedicated files
void bind_channel(pybind11::module_ &);
void bind_control(pybind11::module_ &);


PYBIND11_MODULE(py_sdrlib, m) {
    m.doc() = "C++ SDR module exposed with pybind11";

    bind_channel(m);
    bind_control(m);

}

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "sdrlib/channel.hpp"

namespace py = pybind11;

void bind_channel(pybind11::module_ &m) {

    m.def("apply_cfo",
    [](py::array_t<std::complex<float>> pyarr_in, py::array_t<std::complex<float>> pyarr_out, unsigned int n, float w_offset){
        std::complex<float>* buf_in  = (std::complex<float>*) pyarr_in.request().ptr;
        std::complex<float>* buf_out = (std::complex<float>*) pyarr_out.request().ptr;

        sdrlib::channel::apply_cfo(buf_in, buf_out, n, w_offset);
    },
    py::arg("buf_in"), py::arg("buf_out"), py::arg("n"), py::arg("w_offset"));

}

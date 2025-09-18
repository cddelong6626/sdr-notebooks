
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sdrlib/interpolation.hpp"
#include "sdrlib/types.hpp"

namespace py = pybind11;

void bind_interpolation(pybind11::module_ &m) {

    py::module_ interpolation = m.def_submodule("interpolation", "Interpolation algorithms");

    py::class_<sdrlib::interpolation::CubicFarrowInterpolator>(interpolation, "CubicFarrowInterpolator")
        .def(py::init<>())
        .def("reset", &sdrlib::interpolation::CubicFarrowInterpolator::reset)
        .def("load",
             py::overload_cast<sdrlib::cpx>(&sdrlib::interpolation::CubicFarrowInterpolator::load))
        .def("load",
             [](sdrlib::interpolation::CubicFarrowInterpolator &self,
                py::array_t<sdrlib::cpx> pyarr_in) {
                size_t size = pyarr_in.size();
                py::buffer_info info = pyarr_in.request();
                sdrlib::cpx *buf_in = static_cast<sdrlib::cpx *>(info.ptr);

                self.load(buf_in, size);
             })
        .def("interpolate", &sdrlib::interpolation::CubicFarrowInterpolator::interpolate)
        .def(
            "process",
            [](sdrlib::interpolation::CubicFarrowInterpolator &self,
               py::array_t<sdrlib::cpx> pyarr_in, py::array_t<sdrlib::cpx> pyarr_out,
               float frac_off, int int_off) {
                py::buffer_info info_in = pyarr_in.request();
                py::buffer_info info_out = pyarr_out.request();

                sdrlib::cpx *buf_in = static_cast<sdrlib::cpx *>(info_in.ptr);
                sdrlib::cpx *buf_out = static_cast<sdrlib::cpx *>(info_out.ptr);
                size_t size = static_cast<size_t>(info_in.size);

                self.process(buf_in, buf_out, size, frac_off, int_off);
            },
            py::arg("buffer_in"), py::arg("buffer_out"), py::arg("fractional_offset"),
            py::arg("integer_offset") = 0);
}
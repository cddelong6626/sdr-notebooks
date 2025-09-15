#include "sdrlib/pid_feedback.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_control(pybind11::module_ &m) {

    py::class_<sdrlib::control::PIDFeedback>(m, "PIDFeedback")
        .def(py::init<float, float, float>(), py::arg("K_p") = 0.0f, py::arg("K_i") = 0.0f,
             py::arg("K_d") = 0.0f)
        .def("update", &sdrlib::control::PIDFeedback::update)
        .def("reset", &sdrlib::control::PIDFeedback::reset);
}
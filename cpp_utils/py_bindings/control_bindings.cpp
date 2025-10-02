#include "sdrlib/pid_feedback.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_control(pybind11::module_ &m) {

    py::module_ control = m.def_submodule("control", "Control systems");

    py::class_<sdrlib::control::PIDFeedback>(control, "PIDFeedback")
        .def(py::init<float, float, float>(), py::arg("K_p") = 0.0f, py::arg("K_i") = 0.0f,
             py::arg("K_d") = 0.0f)
        .def("process", &sdrlib::control::PIDFeedback::process)
        .def("reset", &sdrlib::control::PIDFeedback::reset)
        .def_property("K_p", &sdrlib::control::PIDFeedback::get_Kp,
                      &sdrlib::control::PIDFeedback::set_Kp)
        .def_property("K_i", &sdrlib::control::PIDFeedback::get_Ki,
                      &sdrlib::control::PIDFeedback::set_Ki)
        .def_property("K_d", &sdrlib::control::PIDFeedback::get_Kd,
                      &sdrlib::control::PIDFeedback::set_Kd);
}
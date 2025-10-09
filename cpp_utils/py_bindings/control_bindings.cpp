#include "sdrlib/pid_feedback.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_control(pybind11::module_ &m) {

    py::module_ control = m.def_submodule("control", "Control systems");

    // Bindings for PIDFeedback class
    py::class_<sdrlib::control::PIDFeedback>(control, "PIDFeedback")

        // Constructor
        .def(py::init<float, float, float>(), py::arg("K_p") = 0.0f, py::arg("K_i") = 0.0f,
             py::arg("K_d") = 0.0f)

        // Process and reset methods
        .def("process", &sdrlib::control::PIDFeedback::process, py::arg("error"))
        .def("reset", &sdrlib::control::PIDFeedback::reset)

        // Properties for K_p, K_i, K_d
        .def_property("K_p", &sdrlib::control::PIDFeedback::get_Kp,
                      &sdrlib::control::PIDFeedback::set_Kp)
        .def_property("K_i", &sdrlib::control::PIDFeedback::get_Ki,
                      &sdrlib::control::PIDFeedback::set_Ki)
        .def_property("K_d", &sdrlib::control::PIDFeedback::get_Kd,
                      &sdrlib::control::PIDFeedback::set_Kd);
}
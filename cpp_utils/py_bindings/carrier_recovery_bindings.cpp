#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sdrlib/carrier_recovery.hpp"
#include "sdrlib/types.hpp"

namespace py = pybind11;

void bind_carrier_recovery(pybind11::module_ &m) {

    py::module_ carrier_recovery =
        m.def_submodule("carrier_recovery", "Carrier recovery algorithms");

    // Bindings for CostasLoopQPSK class
    py::class_<sdrlib::carrier_recovery::CostasLoopQPSK>(carrier_recovery, "CostasLoopQPSK")

        // Constructor
        .def(py::init<float, int>(), py::arg("loop_bandwidth"),
             py::arg("error_history_size") = 1024)

        // Reset method
        .def("reset", &sdrlib::carrier_recovery::CostasLoopQPSK::reset)

        // Property for loop bandwidth
        .def_property(
            "loop_bw",
            [](sdrlib::carrier_recovery::CostasLoopQPSK &self) { return self.get_loop_bw(); },
            [](sdrlib::carrier_recovery::CostasLoopQPSK &self, float value) {
                self.set_loop_bw(value);
            })

        // Read-only properties for error history and current correction
        // Use lambda to convert std::vector to numpy array
        .def_property_readonly("error_history",
                               [](sdrlib::carrier_recovery::CostasLoopQPSK &self) {
                                   sdrlib::fvec error_history = self.get_error_history();
                                   return py::array_t<float>(error_history.size(),
                                                             error_history.data());
                               })
        .def_property_readonly("correction",
                               &sdrlib::carrier_recovery::CostasLoopQPSK::get_correction)

        // Process a block of samples
        // Use lambda to handle output parameter
        .def(
            "process",
            [](sdrlib::carrier_recovery::CostasLoopQPSK &self, py::array_t<sdrlib::cpx> pyarr_in,
               py::array_t<sdrlib::cpx> pyarr_out) {
                py::buffer_info info_in = pyarr_in.request();
                py::buffer_info info_out = pyarr_out.request();

                sdrlib::cpx *buf_in = static_cast<sdrlib::cpx *>(info_in.ptr);
                sdrlib::cpx *buf_out = static_cast<sdrlib::cpx *>(info_out.ptr);
                size_t size = static_cast<size_t>(info_in.size);

                self.process(buf_in, buf_out, size);
            },
            py::arg("buffer_in"), py::arg("buffer_out")); // n is inferred from array size
}
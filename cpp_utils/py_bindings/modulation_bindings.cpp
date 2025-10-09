#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sdrlib/modulation.hpp"
#include "sdrlib/types.hpp"

namespace py = pybind11;

void bind_modulation(pybind11::module_ &m) {

    py::module_ modulation = m.def_submodule("modulation", "Modulation and demodulation algorithms");

    // Binding for modulate_qpsk function
    // Use a lambda to handle numpy array inputs
    modulation.def(
        "modulate_qpsk",
        [](py::array_t<int> pyarr_in, py::array_t<sdrlib::cpx> pyarr_out) {
            size_t n = pyarr_in.size();
            int *buf_in = static_cast<int *>(pyarr_in.request().ptr);
            sdrlib::cpx *buf_out = static_cast<sdrlib::cpx *>(pyarr_out.request().ptr);

            sdrlib::modulation::modulate_qpsk(buf_in, buf_out, n);
        },
        py::arg("buf_in"), py::arg("buf_out"),
        "Modulate a QPSK signal. Maps 2-bit pairs to complex symbols.\n"
        "Mapping: 00->+1+j, 01->+1-j, 10->-1+j, 11->-1-j");

    // Binding for demodulate_qpsk function
    // Use a lambda to handle numpy array inputs
    modulation.def(
        "demodulate_qpsk",
        [](py::array_t<sdrlib::cpx> pyarr_in, py::array_t<int> pyarr_out) {
            size_t n = pyarr_in.size();
            sdrlib::cpx *buf_in = static_cast<sdrlib::cpx *>(pyarr_in.request().ptr);
            int *buf_out = static_cast<int *>(pyarr_out.request().ptr);

            sdrlib::modulation::demodulate_qpsk(buf_in, buf_out, n);
        },
        py::arg("buf_in"), py::arg("buf_out"),
        "Demodulate a QPSK signal. Maps complex symbols to 2-bit pairs based on quadrant.\n"
        "Mapping: +1+j->00, +1-j->01, -1+j->10, -1-j->11");

    // Binding for optimum_decider_qpsk function
    // Use a lambda to handle numpy array inputs
    modulation.def(
        "optimum_decider_qpsk",
        [](py::array_t<sdrlib::cpx> pyarr_in, py::array_t<int> pyarr_out) {
            size_t n = pyarr_in.size();
            sdrlib::cpx *buf_in = static_cast<sdrlib::cpx *>(pyarr_in.request().ptr);
            int *buf_out = static_cast<int *>(pyarr_out.request().ptr);

            sdrlib::modulation::optimum_decider_qpsk(buf_in, buf_out, n);
        },
        py::arg("buf_in"), py::arg("buf_out"),
        "Optimal decision maker for QPSK symbols based on minimum distance.\n"
        "Each received symbol is compared to the ideal constellation points, and the closest\n"
        "point is selected as the decision.\n"
        "The ideal constellation points are:\n"
        "  +1+j, +1-j, -1+j, -1-j");
}
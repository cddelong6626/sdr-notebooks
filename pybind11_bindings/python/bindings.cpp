#include <pybind11/pybind11.h>

// void bind_filters(pybind11::module_ &);
// void bind_carrier_recovery(pybind11::module_ &);

PYBIND11_MODULE(dsp_cpp, m) {
    m.doc() = "C++ DSP module exposed with pybind11";

    // bind_filters(m);
    // bind_carrier_recovery(m);
}

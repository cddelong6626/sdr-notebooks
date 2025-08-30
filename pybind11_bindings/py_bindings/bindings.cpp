#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <kfr/base.hpp>
#include "sdrlib/carrier_recovery.hpp"

namespace py = pybind11;

namespace py_helpers {

// Convert numpy array to KFR univector
template<typename T>
kfr::univector<T> to_univector(py::array_t<T> arr) {
    auto buf = arr.request();
    return kfr::univector<T>(static_cast<T*>(buf.ptr),
                             static_cast<T*>(buf.ptr) + buf.size);
}

// Convert KFR univector to numpy array
template<typename T>
py::array_t<T> from_univector(const kfr::univector<T>& vec) {
    py::array_t<T> out(vec.size());
    std::copy(vec.begin(), vec.end(), static_cast<T*>(out.request().ptr));
    return out;
}

} // namespace py_helpers


// void bind_filters(pybind11::module_ &);
// void bind_carrier_recovery(pybind11::module_ &);

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(py_sdrlib, m) {
    m.doc() = "C++ SDR module exposed with pybind11";

    m.def("add", &sdrlib::add, "A function that adds two numbers");
    m.def("dot", &sdrlib::dot, "Test");

    // bind_filters(m);
    // bind_carrier_recovery(m);
}

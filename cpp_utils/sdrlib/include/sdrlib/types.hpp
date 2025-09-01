#pragma once 
#include <complex>
#include <kfr/all.hpp>

namespace sdrlib {

// Complex types
using cpx   = std::complex<float>;      // single-precision complex
using cpxd  = std::complex<double>;     // double-precision complex

// Vector types (KFR univector)
using cvec  = kfr::univector<cpx>;      // float complex vector
using cvecd = kfr::univector<cpxd>;     // double complex vector


using fvec  = kfr::univector<float>;    // float vector
using dvec  = kfr::univector<double>;   // double vector

} // namespace sdrlib

#pragma once
#include <kfr/all.hpp>
#include <complex>

namespace sdrlib {

using cvec = kfr::univector<std::complex<float>>;

cvec apply_cfo(const cvec& signal, const float pct_offset);

    
}

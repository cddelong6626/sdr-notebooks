
#pragma once
#include <kfr/all.hpp>
#include <complex>

namespace sdrlib::channel {

void apply_cfo(const std::complex<float>* buf_in, std::complex<float>* buf_out, size_t n, float w_offset);

}

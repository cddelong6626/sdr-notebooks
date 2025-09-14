
#pragma once
#include <kfr/all.hpp>
#include <complex>

#include "sdrlib/types.hpp"

namespace sdrlib::channel {

void apply_cfo(const sdrlib::cpx* buf_in, sdrlib::cpx* buf_out, size_t n, float w_offset);

}

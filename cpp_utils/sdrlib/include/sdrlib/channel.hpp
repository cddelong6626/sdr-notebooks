
#pragma once
#include <complex>
#include <kfr/all.hpp>

#include "sdrlib/types.hpp"

namespace sdrlib::channel {

void apply_cfo(const sdrlib::cpx *buf_in, sdrlib::cpx *buf_out, size_t n, float w_offset);

}

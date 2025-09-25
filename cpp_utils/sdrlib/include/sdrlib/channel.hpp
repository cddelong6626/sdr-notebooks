
#pragma once
#include <complex>
#include <kfr/all.hpp>

#include "sdrlib/types.hpp"

// Channel-related signal processing utilities
namespace sdrlib::channel {

// Applies carrier frequency offset (CFO) correction to the input buffer.
// buf_in:  Pointer to input complex samples.
// buf_out: Pointer to output complex samples (can be same as buf_in for in-place).
// n:       Number of samples.
// w_offset: Frequency offset in radians per sample.
void apply_cfo(const sdrlib::cpx *buf_in, sdrlib::cpx *buf_out, size_t n, float w_offset);

} // namespace sdrlib::channel

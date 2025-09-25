#include <complex>
#include <kfr/base.hpp>

#include "sdrlib/interpolation.hpp"
#include "sdrlib/types.hpp"

namespace sdrlib::interpolation {

void CubicFarrowInterpolator::reset() {
    // Clear the buffer back to zeros
    buffer_idx = 0;
    buffer.ringbuf_write(buffer_idx, 0.0f);
    buffer.ringbuf_write(buffer_idx, 0.0f);
    buffer.ringbuf_write(buffer_idx, 0.0f);
    buffer.ringbuf_write(buffer_idx, 0.0f);
}

void CubicFarrowInterpolator::load(sdrlib::cpx sample) { buffer.ringbuf_write(buffer_idx, sample); }
void CubicFarrowInterpolator::load(sdrlib::cpx *buf_in, size_t size) {
    buffer.ringbuf_write(buffer_idx, buf_in, size);
}

sdrlib::cpx CubicFarrowInterpolator::interpolate(float frac_off, int int_off) {
    // TODO: Assert range for integer_offset
    //  Build sample segment starting at position integer_offset
    sdrlib::cpx *segment_buf = new sdrlib::cpx[N_TAPS];
    buffer.ringbuf_read(buffer_idx, segment_buf, N_TAPS);
    sdrlib::cvec segment_vec = kfr::make_univector(segment_buf, N_TAPS);

    // Use FIR filters to calculate the polynomial coefficients
    sdrlib::cvec c_k = {
        kfr::dotproduct(segment_vec, COEFFS(0)), kfr::dotproduct(segment_vec, COEFFS(1)),
        kfr::dotproduct(segment_vec, COEFFS(2)), kfr::dotproduct(segment_vec, COEFFS(3))};

    // Determine integer offset + fractional offset to approximate
    float mu = int_off + frac_off;

    // Calculate the required powers of mu
    // (Avoid 0^0 issues in case of mu=0 by explicitly setting first term to 1)
    sdrlib::cvec powers = {1.0f, mu, mu * mu, mu * mu * mu};

    // Calculate approximation: c0 + c1*mu + c2*mu^2 + c3*mu^3
    sdrlib::cpx result = kfr::dotproduct(c_k, powers);

    return result;
}

void CubicFarrowInterpolator::process(sdrlib::cpx *buf_in, sdrlib::cpx *buf_out, size_t n,
                                      float frac_off, int int_off) {
    for (size_t i = 0; i < n; ++i) {
        load(buf_in[i]);
        buf_out[i] = interpolate(frac_off, int_off);
    }
}

} // namespace sdrlib::interpolation
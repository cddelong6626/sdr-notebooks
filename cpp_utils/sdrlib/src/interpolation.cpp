#include <kfr/base.hpp>
#include <complex>

#include "sdrlib/types.hpp"
#include "sdrlib/interpolation.hpp"


namespace sdrlib::interpolation {


void CubicFarrowInterpolator::reset() {
    // Clear the buffer back to zeros
    cursor = 0;
    buffer.ringbuf_write(cursor, 0.0f);
    buffer.ringbuf_write(cursor, 0.0f);
    buffer.ringbuf_write(cursor, 0.0f);
    buffer.ringbuf_write(cursor, 0.0f);
}

void CubicFarrowInterpolator::load(sdrlib::cpx sample) {
    buffer.ringbuf_write(cursor, sample);
}
void CubicFarrowInterpolator::load(sdrlib::cpx* buf_in, size_t size) {
    buffer.ringbuf_write(cursor, buf_in, size);
}

sdrlib::cpx CubicFarrowInterpolator::interpolate(float frac_off, int int_off) {
    //TODO: Assert range for integer_offset
    // Build sample segment starting at position integer_offset
    sdrlib::cpx* segment_buf = new sdrlib::cpx[N_TAPS];
    buffer.ringbuf_read(cursor, segment_buf, N_TAPS);
    sdrlib::cvec segment_vec = kfr::make_univector(segment_buf, N_TAPS);

    std::cout << segment_vec[0] << "," << segment_vec[1] << "," << segment_vec[2] << "," << segment_vec[3] << std::endl;

    // Use FIR filters to calculate the polynomial coefficients
    auto c_k = COEFFS * segment_vec;
    
    // Determine index + fractional offset to approximate
    frac_off -= int_off;

    // Calculate the needed powers of mu
    auto powers_expr = kfr::pow(frac_off, kfr::counter(0, 1));
    cvec powers = kfr::render(powers_expr, N_TAPS);

    // Calculate approximation: c0 + c1*mu + c2*mu^2 + c3*mu^3
    cpx result = kfr::dotproduct(c_k, powers);

    return result;
}

void CubicFarrowInterpolator::process(sdrlib::cpx* buf_in, sdrlib::cpx* buf_out, size_t n, float frac_off, int int_off) {
    load(buf_in, n);
    for (size_t i = 0; i < n; ++i) {
        buf_out[i] = interpolate(frac_off, int_off);
    }
}


} // namespace sdrlib::interpolation
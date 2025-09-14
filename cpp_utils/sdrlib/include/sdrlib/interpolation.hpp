
#pragma once
#include <kfr/all.hpp>
#include <complex>
#include <sdrlib/types.hpp>


namespace sdrlib::interpolation {

class CubicFarrowInterpolator {
private:
    // Ring buffer to store previous inputs
    sdrlib::cvec buffer {0.0f, 0.0f, 0.0f, 0.0f};
    size_t cursor = 0;

public:
    // This class implements 3rd order Lagrange interpolation
    const char ORDER = 3;
    const char N_TAPS = ORDER + 1;

    // Lagrange basis coefficients
    const kfr::tensor<cpx, 2> COEFFS {
        {0, 1, 0, 0},
        {-1/3, -1/2, 1, -1/6},
        {1/2, -1, 1/2, 0},
        {-1/6, 1/2, -1/2, 1/6}
    };

    
    CubicFarrowInterpolator() {};

    void reset();

    void load(sdrlib::cpx sample);
    void load(sdrlib::cpx* samples, size_t size);

    sdrlib::cpx interpolate(float mu, int int_off = 0);

    void process(sdrlib::cpx* buf_in, sdrlib::cpx* buf_out, size_t n, float frac_off, int int_off = 0);

};


} // namespace sdrlib::interpolation
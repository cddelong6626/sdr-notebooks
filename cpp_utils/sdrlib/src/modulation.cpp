
#include <complex>

#include "sdrlib/modulation.hpp"
#include "sdrlib/types.hpp"

#include <iostream>

namespace sdrlib::modulation {

// Function to modulate a QPSK signal
void modulate_qpsk(const int *buf_in, sdrlib::cpx *buf_out, size_t n) {
    /**
     * Modulates a QPSK signal by mapping each pair of bits to a complex symbol.
     * The mapping is as follows:
     *   00 -> +1+j
     *   01 -> +1-j
     *   10 -> -1+j
     *   11 -> -1-j
     **/

    // Map each 2-bit pair to a complex symbol
    for (size_t i = 0; i < n; ++i) {
        switch (buf_in[i] & 0b11) {
        case 0b00:
            buf_out[i] = sdrlib::cpx(+1.0f, +1.0f); // 00 -> +1+j
            break;
        case 0b01:
            buf_out[i] = sdrlib::cpx(+1.0f, -1.0f); // 01 -> +1-j
            break;
        case 0b10:
            buf_out[i] = sdrlib::cpx(-1.0f, +1.0f); // 10 -> -1+j
            break;
        case 0b11:
            buf_out[i] = sdrlib::cpx(-1.0f, -1.0f); // 11 -> -1-j
            break;
        default:
            buf_out[i] = sdrlib::cpx(0.0f, 0.0f);
        }
    }
}

// Function to demodulate a QPSK signal
void demodulate_qpsk(const sdrlib::cpx *buf_in, int *buf_out, size_t n) {
    /**
     * Demodulates a QPSK signal by mapping each complex symbol to 2 bits.
     * The mapping is based on the quadrant of the complex plane:
     *   +1+j -> 00
     *   +1-j -> 01
     *   -1+j -> 10
     *   -1-j -> 11
     **/

    // Map each complex symbol to 2 bits based on quadrant using bit masking
    for (size_t i = 0; i < n; ++i) {
        int bits = 0;
        if (buf_in[i].real() < 0)
            bits |= 0b10;
        if (buf_in[i].imag() < 0)
            bits |= 0b01;
        buf_out[i] = bits;
    }
}

// Optimal decision maker for QPSK symbols (minimum distance)
void optimum_decider_qpsk(const sdrlib::cpx *buf_in, int *buf_out, size_t n) {
    /**
     * Implements an optimal decision maker for QPSK symbols based on minimum distance.
     * The closest constellation point is determined using the sign of the real and imaginary parts.
     * The ideal constellation points are:
     *   +1+j, +1-j, -1+j, -1-j
     **/

    // Map each complex symbol to 2 bits based on minimum distance
    for (size_t i = 0; i < n; ++i) {
        int bits = 0;
        if (buf_in[i].real() < 0)
            bits |= 0b10;
        if (buf_in[i].imag() < 0)
            bits |= 0b01;
        buf_out[i] = bits;
    }
}

} // namespace sdrlib::modulation
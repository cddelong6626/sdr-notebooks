
#pragma once
#include <complex>
#include <kfr/all.hpp>

#include "sdrlib/types.hpp"

// Modulation-related signal processing utilities
namespace sdrlib::modulation {

// Modulate a QPSK signal
void modulate_qpsk(const int *buf_in, sdrlib::cpx *buf_out, size_t n);

// Demodulate a QPSK signal
void demodulate_qpsk(const sdrlib::cpx *buf_in, int *buf_out, size_t n);

// Optimal decision maker for QPSK symbols (minimum distance)
void optimum_decider_qpsk(const sdrlib::cpx *buf_in, int *buf_out, size_t n);

} // namespace sdrlib::modulation
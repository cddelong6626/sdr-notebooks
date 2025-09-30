#include "sdrlib/carrier_recovery.hpp"
#include "sdrlib/pid_feedback.hpp"
#include "sdrlib/types.hpp"
#include <cmath>
#include <kfr/base.hpp>
#include <kfr/dft.hpp>
#include <kfr/dsp.hpp>

namespace sdrlib::carrier_recovery {

CostasLoopQPSK::CostasLoopQPSK(float loop_bw) {
    set_loop_bw(loop_bw);
    correction = 0.0f;

    error_history_idx = 0;
}

void CostasLoopQPSK::reset() {
    correction = 0.0f;
    std::fill(error_history.begin(), error_history.end(), 0.0f);
    error_history_idx = 0;

    controller.reset();
}

void CostasLoopQPSK::set_loop_bw(float value) {
    // The following equations were derived here: https://john-gentile.com/kb/dsp/PI_filter.html
    loop_bw = value;

    float damping_factor = 0.707;
    float normalized_freq = loop_bw; // assuming sample rate = 1

    float alpha = 1.0f - 2.0f * damping_factor * damping_factor;
    float scaled_bw = normalized_freq / std::sqrt(alpha + std::sqrt(alpha * alpha + 1.0f));
    float K_d = 1.0f;
    float K_p = 2.0f * damping_factor * scaled_bw / K_d;
    float K_i = (scaled_bw * scaled_bw) / K_d;

    controller.set_Kp(K_p);
    controller.set_Ki(K_i);
}

float CostasLoopQPSK::get_loop_bw() const { return loop_bw; }

float CostasLoopQPSK::get_correction() const { return correction; }

sdrlib::fvec CostasLoopQPSK::get_error_history() const { return error_history; }

void CostasLoopQPSK::process_sample(sdrlib::cpx &symbol_in, sdrlib::cpx &symbol_out) {

    // Rotate the input signal by the current correction estimate
    symbol_out = symbol_in * kfr::cexp(sdrlib::cpx(0.0f, -correction));

    // Make a decision about the current symbol value and use that as a reference to find the error
    float I = symbol_out.real();
    float Q = symbol_out.imag();

    sdrlib::cpx ref = sdrlib::cpx(std::copysign(1.0f, I), std::copysign(1.0f, Q));
    float e = kfr::carg(symbol_out * kfr::cconj(ref));
    error_history.ringbuf_write(error_history_idx, e);

    // Update correction estimate
    correction += controller.update(e);
}

void CostasLoopQPSK::process(sdrlib::cpx *buf_in, sdrlib::cpx *buf_out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        process_sample(buf_in[i], buf_out[i]);
    }
}

} // namespace sdrlib::carrier_recovery
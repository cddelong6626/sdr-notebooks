
#include "sdrlib/pid_feedback.hpp"
#include <complex>
#include <kfr/all.hpp>

namespace sdrlib::control {

PIDFeedback::PIDFeedback(float Kp, float Ki, float Kd) : K_p(Kp), K_i(Ki), K_d(Kd) {}

float PIDFeedback::update(float e) {
    // Calculate update to input value
    sum_e += e;
    float result = (K_i * sum_e) + (K_p * e) + (K_d * (e - prev_e));

    // Update previous error
    prev_e = e;

    return result;
}

void PIDFeedback::reset() {
    sum_e = 0.0;
    prev_e = 0.0;
}

} // namespace sdrlib::control

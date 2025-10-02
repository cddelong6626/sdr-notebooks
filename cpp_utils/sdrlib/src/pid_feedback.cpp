
#include "sdrlib/pid_feedback.hpp"
#include <complex>
#include <kfr/all.hpp>

namespace sdrlib::control {

PIDFeedback::PIDFeedback(float Kp, float Ki, float Kd) : K_p(Kp), K_i(Ki), K_d(Kd) {}

// Process method
float PIDFeedback::process(float e) {
    // Calculate update to input value
    sum_e += e;
    float result = (K_i * sum_e) + (K_p * e) + (K_d * (e - prev_e));

    // Update previous error
    prev_e = e;

    return result;
}

// Reset method
void PIDFeedback::reset() {
    sum_e = 0.0;
    prev_e = 0.0;
}

// Getter methods
float PIDFeedback::get_Kp() const { return K_p; }

float PIDFeedback::get_Ki() const { return K_i; }

float PIDFeedback::get_Kd() const { return K_d; }

// Setter methods
void PIDFeedback::set_Kp(float Kp) { K_p = Kp; }

void PIDFeedback::set_Ki(float Ki) { K_i = Ki; }

void PIDFeedback::set_Kd(float Kd) { K_d = Kd; }

} // namespace sdrlib::control

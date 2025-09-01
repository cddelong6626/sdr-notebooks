
#include <kfr/all.hpp>
#include <complex>
#include "sdrlib/pid_feedback.hpp"

namespace sdrlib::control {


PIDFeedback::PIDFeedback(float K_p, float K_i, float K_d)
        : K_p_(K_p), K_i_(K_i), K_d_(K_d) {}

float PIDFeedback::update(float e) {
    // Calculate update to input value
    sum_e += e;
    float result = 
        (K_i_ * sum_e) + 
        (K_p_ * e) + 
        (K_d_ * (e - prev_e));

    // Update previous error
    prev_e = e;

    return result;
}


void PIDFeedback::reset() {
    sum_e = 0.0;
    prev_e = 0.0;
}


}



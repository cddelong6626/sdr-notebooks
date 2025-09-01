
#pragma once
#include <kfr/all.hpp>
#include <complex>

namespace sdrlib::control {

class PIDFeedback {
private:
    float K_p_;
    float K_i_;
    float K_d_;

    float sum_e = 0.0f;
    float prev_e = 0.0f;

public:
    PIDFeedback(float K_p = 0.0f, float K_i = 0.0f, float K_d = 0.0f);

    float update(float e);

    void reset();
};


}

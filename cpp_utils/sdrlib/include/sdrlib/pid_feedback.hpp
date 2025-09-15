
#pragma once
#include <complex>
#include <kfr/all.hpp>

namespace sdrlib::control {

class PIDFeedback {
  private:
    float sum_e = 0.0f;
    float prev_e = 0.0f;

  public:
    float K_p = 0.0f;
    float K_i = 0.0f;
    float K_d = 0.0f;

    PIDFeedback(float Kp, float Ki, float Kd);

    float update(float e);

    void reset();
};

} // namespace sdrlib::control


#pragma once
#include <complex>
#include <kfr/all.hpp>

namespace sdrlib::control {

class PIDFeedback {
  private:
    // State variables
    float sum_e = 0.0f;
    float prev_e = 0.0f;

    // PID coefficients
    float K_p = 0.0f;
    float K_i = 0.0f;
    float K_d = 0.0f;

  public:
    PIDFeedback(float Kp, float Ki, float Kd);

    // Update method
    float update(float e);

    // Reset method
    void reset();

    // Getter methods
    float get_Kp() const;
    float get_Ki() const;
    float get_Kd() const;

    // Setter methods
    void set_Kp(float Kp);
    void set_Ki(float Ki);
    void set_Kd(float Kd);
};

} // namespace sdrlib::control

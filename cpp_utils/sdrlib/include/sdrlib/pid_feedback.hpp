
#pragma once
#include <complex>
#include <kfr/all.hpp>

namespace sdrlib::control {

/**
 * @class PIDFeedback
 * @brief Simple PID feedback controller used for loop control tasks.
 *
 * This class implements a proportional-integral-derivative (PID) controller with
 * a minimal API: construction with gains, update to compute the control output
 * given a new error sample, reset to clear internal state, and getters/setters
 * for each gain.
 *
 * @note This class is not thread-safe.
 */
class PIDFeedback {
  private:
    /**
     * @brief Running integral term (accumulated error).
     *
     * Used to compute the integral component of the control output.
     */
    float sum_e = 0.0f;

    /**
     * @brief Previous error sample.
     *
     * Used to compute the derivative component of the control output.
     */
    float prev_e = 0.0f;

    /**
     * @brief Proportional gain.
     */
    float K_p = 0.0f;

    /**
     * @brief Integral gain.
     */
    float K_i = 0.0f;

    /**
     * @brief Derivative gain.
     */
    float K_d = 0.0f;

  public:
    /**
     * @brief Constructs a PIDFeedback controller with the specified gains.
     * @param Kp Proportional gain.
     * @param Ki Integral gain.
     * @param Kd Derivative gain.
     */
    PIDFeedback(float Kp, float Ki, float Kd);

    /**
     * @brief Update the controller with a new error sample and obtain control output.
     * @param e The current error value.
     * @return The computed control value: Kp*e + Ki*integral + Kd*derivative.
     */
    float process(float e);

    /**
     * @brief Reset internal state (integrator and previous error).
     *
     * Clears accumulated state so the controller starts fresh.
     */
    void reset();

    /**
     * @brief Get the proportional gain.
     * @return Current Kp value.
     */
    float get_Kp() const;

    /**
     * @brief Get the integral gain.
     * @return Current Ki value.
     */
    float get_Ki() const;

    /**
     * @brief Get the derivative gain.
     * @return Current Kd value.
     */
    float get_Kd() const;

    /**
     * @brief Set the proportional gain.
     * @param Kp New proportional gain.
     */
    void set_Kp(float Kp);

    /**
     * @brief Set the integral gain.
     * @param Ki New integral gain.
     */
    void set_Ki(float Ki);

    /**
     * @brief Set the derivative gain.
     * @param Kd New derivative gain.
     */
    void set_Kd(float Kd);
};

} // namespace sdrlib::control

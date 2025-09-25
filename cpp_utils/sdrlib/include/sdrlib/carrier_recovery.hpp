
#pragma once
#include "sdrlib/pid_feedback.hpp"
#include "sdrlib/types.hpp"

// Carrier recovery-related signal processing utilities
namespace sdrlib::carrier_recovery {

/**
 * @class CostasLoopQPSK
 * @brief Implements a QPSK Costas loop for carrier recovery in digital communication systems.
 *
 * This class provides methods to perform carrier phase recovery using a Costas loop,
 * specifically designed for Quadrature Phase Shift Keying (QPSK) signals. It maintains
 * an internal error history buffer for analysis and debugging purposes.
 *
 * @note The class is not thread-safe.
 */
class CostasLoopQPSK {
  private:
    /**
     * @brief Loop bandwidth parameter controlling the response speed and noise rejection.
     */
    float loop_bw;

    /**
     * @brief Current correction value applied to the carrier phase.
     */
    float correction = 0.0f;

    /**
     * @brief Circular buffer storing recent error values for monitoring loop performance.
     */
    sdrlib::fvec error_history = sdrlib::fvec(10000, 0.0f);

    /**
     * @brief Index for the next error value to be written in the error history buffer.
     */
    size_t error_history_idx = 0;

    /**
     * @brief Internal PID controller used to compute phase corrections.
     */
    sdrlib::control::PIDFeedback controller = sdrlib::control::PIDFeedback(0.0f, 0.0f, 0.0f);

  public:
    /**
     * @brief Constructs a CostasLoopQPSK object with the specified loop bandwidth.
     * @param loop_bw The initial loop bandwidth.
     */
    CostasLoopQPSK(float loop_bw);

    /**
     * @brief Retrieves the current correction value.
     * @return The current phase correction applied by the loop.
     */
    float get_correction() const;

    /**
     * @brief Sets the loop bandwidth.
     * @param value The new loop bandwidth value.
     */
    void set_loop_bw(float value);

    /**
     * @brief Gets the current loop bandwidth.
     * @return The current loop bandwidth.
     */
    float get_loop_bw() const;

    /**
     * @brief Resets the internal state of the Costas loop, including error history and correction.
     */
    void reset();

    /**
     * @brief Retrieves a copy of the error history buffer.
     * @return A vector containing recent error values.
     */
    sdrlib::fvec get_error_history() const;

    /**
     * @brief Processes a single complex input sample and outputs the corrected sample.
     * @param sample_in The input complex sample.
     * @param sample_out The output complex sample after carrier correction.
     */
    void process_sample(sdrlib::cpx &symbol_in, sdrlib::cpx &symbol_out);

    /**
     * @brief Processes a buffer of complex samples, applying carrier recovery to each.
     * @param buf_in Pointer to the input buffer of complex samples.
     * @param buf_out Pointer to the output buffer for corrected samples.
     * @param n Number of samples to process.
     */
    void process(sdrlib::cpx *buf_in, sdrlib::cpx *buf_out, size_t n);
};

} // namespace sdrlib::carrier_recovery

#pragma once
#include <complex>
#include <kfr/all.hpp>
#include <sdrlib/types.hpp>

// Interpolation-related signal processing utilities
namespace sdrlib::interpolation {

/**
 * @brief Cubic Farrow interpolator implementing 3rd-order Lagrange interpolation.
 *
 * This class provides sample-rate interpolation using a 3rd-order Farrow/Lagrange
 * structure. It maintains a small ring buffer of recent input samples and exposes
 * methods to load new samples and compute interpolated outputs at fractional
 * sample positions.
 *
 * Notes:
 * - The interpolator is intended for real-time streaming usage; calls to load()
 *   advance the internal buffer/buffer_idx.
 * - The implementation is not thread-safe.
 */
class CubicFarrowInterpolator {
  private:
    /// Ring buffer to store previous 4 input samples for interpolation
    sdrlib::cvec buffer{0.0f, 0.0f, 0.0f, 0.0f};
    /// Current position in the ring buffer
    size_t buffer_idx = 0;

  public:
    /// Interpolator order (3rd order)
    const char ORDER = 3;
    /// Number of taps (ORDER + 1)
    const char N_TAPS = ORDER + 1;

    /**
     * @brief Lagrange basis coefficients for cubic interpolation.
     *
     * COEFFS[row][col] correspond to polynomial coefficients used by the Farrow
     * structure to compute interpolated values from the four stored samples.
     */
    const kfr::tensor<sdrlib::cpx, 2> COEFFS{{0.0f, 1.0f, 0.0f, 0.0f},
                                             {-1.0f / 3, -1.0f / 2, 1.0f, -1.0f / 6},
                                             {1.0f / 2, -1.0f, 1.0f / 2, 0.0f},
                                             {-1.0f / 6, 1.0f / 2, -1.0f / 2, 1.0f / 6}};

    /**
     * @brief Construct a new CubicFarrowInterpolator object.
     */
    CubicFarrowInterpolator(){};

    /**
     * @brief Get the internal interpolator buffer.
     * @return const sdrlib::cvec The interpolator buffer.
     */
    sdrlib::cvec get_buffer() const;

    /**
     * @brief Reset the internal state of the interpolator.
     *
     * Clears the ring buffer and resets the internal buffer_idx to its initial state.
     */
    void reset();

    /**
     * @brief Load a single input sample into the ring buffer.
     * @param sample The complex sample to append to the internal buffer.
     *
     * Calling load advances the internal buffer_idx; subsequent calls will overwrite
     * the oldest sample in the ring buffer when full.
     */
    void load(sdrlib::cpx sample);

    /**
     * @brief Load multiple input samples into the ring buffer.
     * @param samples Pointer to an array of complex samples to load.
     * @param size Number of samples pointed to by samples.
     *
     * This convenience method repeatedly calls load(sample) for each element in
     * the provided array.
     */
    void load(sdrlib::cpx *samples, size_t size);

    /**
     * @brief Interpolate at a fractional position mu with optional integer offset.
     * @param mu Fractional interpolation position (expected in [0, 1)).
     * @param int_off Optional integer sample offset to apply before interpolation.
     * @return sdrlib::cpx The interpolated complex sample.
     *
     * The method evaluates the Farrow/Lagrange polynomial using the contents of
     * the internal ring buffer. mu represents the fractional distance between
     * neighboring integer samples.
     */
    sdrlib::cpx interpolate(float mu, int int_off = 0);

    /**
     * @brief Process n samples from buf_in to buf_out with given fractional and integer offsets.
     * @param buf_in Pointer to input buffer of complex samples.
     * @param buf_out Pointer to output buffer where interpolated samples will be written.
     * @param n Number of output samples to produce.
     * @param frac_off Fractional offset (mu) to use for interpolation.
     * @param int_off Optional integer offset applied to interpolation phase.
     *
     * This routine applies interpolation across a block of input samples, producing
     * n interpolated outputs written to buf_out. It advances the internal buffer
     * as it consumes buf_in.
     */
    void process(sdrlib::cpx *buf_in, sdrlib::cpx *buf_out, size_t n, float frac_off,
                 int int_off = 0);
};

} // namespace sdrlib::interpolation
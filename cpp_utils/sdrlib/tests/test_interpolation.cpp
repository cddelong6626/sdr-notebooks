#include "sdrlib/interpolation.hpp"
#include <complex>
#include <gtest/gtest.h>

using namespace sdrlib;
using namespace interpolation;

class CubicFarrowInterpolatorTest : public ::testing::Test {
  protected:
    CubicFarrowInterpolator interp;

    void TearDown() override { interp.reset(); }
};

TEST_F(CubicFarrowInterpolatorTest, InterpolateZeroOffset) {
    cpx samples[4] = {{1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}, {4.0f, 0.0f}};

    // Load samples using the bulk load method
    interp.load(samples, 4);

    float frac_off = 0;
    int int_off = 0;
    cpx result = interp.interpolate(frac_off, int_off);

    EXPECT_NEAR(result.real(), 2.0f, 1e-4);
    EXPECT_NEAR(result.imag(), 0.0f, 1e-4);
}

TEST_F(CubicFarrowInterpolatorTest, InterpolateHalfway) {
    cpx samples[4] = {{1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}, {4.0f, 0.0f}};

    // Load samples one by one to test single-sample load
    for (int i = 0; i < 4; ++i) {
        interp.load(samples[i]);
    }

    float frac_off = 0.5f;
    int int_off = 1;
    cpx result = interp.interpolate(frac_off, int_off);

    EXPECT_NEAR(result.real(), 3.5f, 1e-4);
    EXPECT_NEAR(result.imag(), 0.0f, 1e-4);
}

TEST_F(CubicFarrowInterpolatorTest, InterpolateWithComplexInput) {
    cpx samples[4] = {{1.0f, 1.0f}, {2.0f, 2.0f}, {3.0f, 3.0f}, {4.0f, 4.0f}};
    interp.load(samples, 4);

    float frac_off = 0.25f;
    int int_off = 0;
    cpx result = interp.interpolate(frac_off, int_off);

    EXPECT_NEAR(result.imag(), result.real(), 1e-4);
}

TEST_F(CubicFarrowInterpolatorTest, InterpolateNegativeOffset) {
    cpx samples[4] = {{1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}, {4.0f, 0.0f}};
    interp.load(samples, 4);

    float frac_off = -0.5f;
    int int_off = 0;
    cpx result = interp.interpolate(frac_off, int_off);

    EXPECT_NEAR(result.real(), 1.5f, 1e-4);
    EXPECT_NEAR(result.imag(), 0.0f, 1e-4);
}

TEST_F(CubicFarrowInterpolatorTest, ProcessMultipleSamples_AllOutputs) {
    cpx input_samples[6] = {{1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f},
                            {4.0f, 0.0f}, {5.0f, 0.0f}, {6.0f, 0.0f}};
    cpx output_samples[6];

    float frac_off = 0.5f;
    int int_off = 1;
    interp.process(input_samples, output_samples, 6, frac_off, int_off);

    // For i < 3, the buffer is not fully filled, so output is not very meaningful.
    // We'll just check that the outputs are finite. For i >= 3, we can check expected values.

    // For output_samples[3], buffer contains {1,2,3,4}, int_off=1, frac_off=0.5
    // As in previous test, expected real = 3.5
    EXPECT_NEAR(output_samples[3].real(), 3.5f, 1e-4);
    EXPECT_NEAR(output_samples[3].imag(), 0.0f, 1e-4);

    // For output_samples[4], buffer contains {2,3,4,5}
    EXPECT_NEAR(output_samples[4].real(), 4.5f, 1e-4);
    EXPECT_NEAR(output_samples[4].imag(), 0.0f, 1e-4);

    // For output_samples[5], buffer contains {3,4,5,6}
    EXPECT_NEAR(output_samples[5].real(), 5.5f, 1e-4);
    EXPECT_NEAR(output_samples[5].imag(), 0.0f, 1e-4);

    // Check that the first three outputs are finite (since buffer is not fully filled)
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(std::isfinite(output_samples[i].real()));
        EXPECT_TRUE(std::isfinite(output_samples[i].imag()));
    }
}
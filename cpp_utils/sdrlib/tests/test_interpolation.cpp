#include "sdrlib/interpolation.hpp"
#include <complex>
#include <gtest/gtest.h>

using namespace sdrlib::interpolation;

class CubicFarrowInterpolatorTest : public ::testing::Test {
  protected:
    CubicFarrowInterpolator interp;

    void TearDown() override { interp.reset(); }
};

TEST_F(CubicFarrowInterpolatorTest, InterpolateZeroOffset) {
    sdrlib::cpx samples[4] = {{1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}, {4.0f, 0.0f}};
    interp.load(samples, 4);

    float frac_off = 0;
    int int_off = 0;
    sdrlib::cpx result = interp.interpolate(frac_off, int_off);

    EXPECT_NEAR(result.real(), 2.0f, 1e-4);
    EXPECT_NEAR(result.imag(), 0.0f, 1e-4);
}

TEST_F(CubicFarrowInterpolatorTest, InterpolateHalfway) {
    sdrlib::cpx samples[4] = {{1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}, {4.0f, 0.0f}};
    interp.load(samples, 4);

    float frac_off = 0.5f;
    int int_off = 0;
    sdrlib::cpx result = interp.interpolate(frac_off, int_off);

    EXPECT_NEAR(result.real(), 2.5f, 1e-4);
    EXPECT_NEAR(result.imag(), 0.0f, 1e-4);
}

TEST_F(CubicFarrowInterpolatorTest, InterpolateWithComplexInput) {
    sdrlib::cpx samples[4] = {{1.0f, 1.0f}, {2.0f, 2.0f}, {3.0f, 3.0f}, {4.0f, 4.0f}};
    interp.load(samples, 4);

    float frac_off = 0.25f;
    int int_off = 0;
    sdrlib::cpx result = interp.interpolate(frac_off, int_off);

    EXPECT_NEAR(result.imag(), result.real(), 1e-4);
}

TEST_F(CubicFarrowInterpolatorTest, InterpolateNegativeOffset) {
    sdrlib::cpx samples[4] = {{1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}, {4.0f, 0.0f}};
    interp.load(samples, 4);

    float frac_off = -0.5f;
    int int_off = 0;
    sdrlib::cpx result = interp.interpolate(frac_off, int_off);

    EXPECT_NEAR(result.real(), 1.5f, 1e-4);
    EXPECT_NEAR(result.imag(), 0.0f, 1e-4);
}
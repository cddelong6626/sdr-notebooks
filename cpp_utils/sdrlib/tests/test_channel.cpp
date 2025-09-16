#include "sdrlib/channel.hpp"
#include "sdrlib/types.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace sdrlib;

// Helper function to compare two complex vectors
void expect_cvec_near(const cpx *a, const cpx *b, size_t n, float tol = 1e-4f) {
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(a[i].real(), b[i].real(), tol) << "at index " << i;
        EXPECT_NEAR(a[i].imag(), b[i].imag(), tol) << "at index " << i;
    }
}

TEST(ChannelTest, ApplyCfoZeroOffset) {
    constexpr size_t N = 8;
    cpx input[N];
    cpx output[N];
    for (size_t i = 0; i < N; ++i)
        input[i] = cpx(float(i), -float(i));

    channel::apply_cfo(input, output, N, 0.0f);

    expect_cvec_near(input, output, N);
}

TEST(ChannelTest, ApplyCfoPiOffset) {
    constexpr size_t N = 4;
    cpx input[N] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    cpx expected[N];
    float w_offset = float(M_PI);

    // e^(j*n*pi) alternates between 1, -1, 1, -1
    for (size_t n = 0; n < N; ++n) {
        float phase = n * w_offset;
        cpx rot = std::polar(1.0f, phase);
        expected[n] = input[n] * rot;
    }

    cpx output[N];
    channel::apply_cfo(input, output, N, w_offset);

    expect_cvec_near(expected, output, N);
}

TEST(ChannelTest, ApplyCfoKnownRotation) {
    constexpr size_t N = 3;
    cpx input[N] = {{1, 0}, {1, 0}, {1, 0}};
    cpx expected[N];
    float w_offset = float(M_PI_2); // 90 degrees per sample

    for (size_t n = 0; n < N; ++n) {
        float phase = n * w_offset;
        cpx rot = std::polar(1.0f, phase);
        expected[n] = input[n] * rot;
    }

    cpx output[N];
    channel::apply_cfo(input, output, N, w_offset);

    expect_cvec_near(expected, output, N);
}

TEST(ChannelTest, ApplyCfoEmptyInput) {
    cpx *input = nullptr;
    cpx *output = nullptr;
    // Should not crash
    channel::apply_cfo(input, output, 0, 1.0f);
}

TEST(ChannelTest, ApplyCfoLargeBuffer) {
    constexpr size_t N = 10000;
    std::vector<cpx> input(N), expected(N), output(N);
    float w_offset = 0.001f;

    for (size_t n = 0; n < N; ++n) {
        input[n] = cpx(float(n), -float(n));
        float phase = n * w_offset;
        cpx rot = std::polar(1.0f, phase);
        expected[n] = input[n] * rot;
    }

    channel::apply_cfo(input.data(), output.data(), N, w_offset);

    expect_cvec_near(expected.data(), output.data(), N, 2e-3f);
}

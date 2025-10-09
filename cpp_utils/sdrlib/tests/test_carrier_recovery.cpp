#include "sdrlib/carrier_recovery.hpp"
#include "sdrlib/types.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace sdrlib;
using namespace carrier_recovery;

// Helper function to create a QPSK symbol
// TODO: Update to use sdrlib::modulation once available
cpx create_qpsk_symbol(int bits) {
    switch (bits & 0b11) {
    case 0:
        return cpx(1.0f, 1.0f); // 00 -> +1+j
    case 1:
        return cpx(1.0f, -1.0f); // 01 -> +1-j
    case 2:
        return cpx(-1.0f, -1.0f); // 10 -> -1-j
    case 3:
        return cpx(-1.0f, 1.0f); // 11 -> -1+j
    default:
        return cpx(0.0f, 0.0f);
    }
}

// Helper function to apply phase rotation
cpx apply_phase_rotation(cpx symbol, float phase) { return symbol * std::polar(1.0f, phase); }

class CostasLoopQPSKTest : public ::testing::Test {
  protected:
    static constexpr float DEFAULT_LOOP_BW = 1.0f / 20.0f;
    CostasLoopQPSK costas{DEFAULT_LOOP_BW};

    void SetUp() override { costas.reset(); }
};

TEST_F(CostasLoopQPSKTest, InitialState) {
    EXPECT_FLOAT_EQ(costas.get_loop_bw(), DEFAULT_LOOP_BW);
    EXPECT_FLOAT_EQ(costas.get_correction(), 0.0f);

    auto error_history = costas.get_error_history();
    for (const auto &e : error_history) {
        EXPECT_FLOAT_EQ(e, 0.0f);
    }
}

TEST_F(CostasLoopQPSKTest, ZeroPhaseError) {
    cpx input_symbol = create_qpsk_symbol(0); // +1+j
    cpx output_symbol;

    costas.process_sample(input_symbol, output_symbol);

    // With no phase error, output should match input
    EXPECT_NEAR(output_symbol.real(), input_symbol.real(), 1e-6);
    EXPECT_NEAR(output_symbol.imag(), input_symbol.imag(), 1e-6);
}

TEST_F(CostasLoopQPSKTest, ConstantPhaseError) {
    constexpr float phase_error = M_PI / 8; // 22.5 degrees
    constexpr size_t num_symbols = 100;

    // Process multiple symbols with constant phase error
    for (size_t i = 0; i < num_symbols; ++i) {
        cpx input_symbol = create_qpsk_symbol(i % 4);
        cpx rotated_symbol = apply_phase_rotation(input_symbol, phase_error);
        cpx output_symbol;

        costas.process_sample(rotated_symbol, output_symbol);

        // After convergence, output should be closer to original
        if (i > 50) { // Allow time for convergence
            float error_magnitude = std::abs(output_symbol - input_symbol);
            EXPECT_LT(error_magnitude, 0.5f);
        }
    }

    // Loop should have tracked and reduced the phase error
    EXPECT_NE(costas.get_correction(), 0.0f);
}

TEST_F(CostasLoopQPSKTest, AllQPSKConstellationPoints) {
    constexpr float phase_error = M_PI / 6; // 30 degrees

    for (int bits = 0; bits < 4; ++bits) {
        costas.reset();
        cpx input_symbol = create_qpsk_symbol(bits);
        cpx rotated_symbol = apply_phase_rotation(input_symbol, phase_error);
        cpx output_symbol;

        // Process the same symbol multiple times to allow convergence
        for (int i = 0; i < 50; ++i) {
            costas.process_sample(rotated_symbol, output_symbol);
        }

        // Check that the correction brings the symbol closer to original
        float corrected_error = std::abs(output_symbol - input_symbol);
        float uncorrected_error = std::abs(rotated_symbol - input_symbol);
        EXPECT_LT(corrected_error, uncorrected_error);
    }
}

TEST_F(CostasLoopQPSKTest, Reset) {
    // Apply some phase error to change the internal state
    cpx input_symbol = create_qpsk_symbol(0);
    cpx rotated_symbol = apply_phase_rotation(input_symbol, M_PI / 4);
    cpx output_symbol;

    for (int i = 0; i < 20; ++i) {
        costas.process_sample(rotated_symbol, output_symbol);
    }

    // State should be non-zero after processing
    EXPECT_NE(costas.get_correction(), 0.0f);

    // Reset and verify state is cleared
    costas.reset();
    EXPECT_FLOAT_EQ(costas.get_correction(), 0.0f);

    auto error_history = costas.get_error_history();
    for (const auto &e : error_history) {
        EXPECT_FLOAT_EQ(e, 0.0f);
    }
}

TEST_F(CostasLoopQPSKTest, SmallLoopBandwidth) {
    CostasLoopQPSK slow_costas(1.0f / 100.0f); // Very small bandwidth

    EXPECT_FLOAT_EQ(slow_costas.get_loop_bw(), 1.0f / 100.0f);

    cpx input_symbol = create_qpsk_symbol(0);
    cpx rotated_symbol = apply_phase_rotation(input_symbol, M_PI / 8);
    cpx output_symbol;

    // Should converge more slowly
    slow_costas.process_sample(rotated_symbol, output_symbol);
    float correction_after_one = slow_costas.get_correction();

    // Compare with default bandwidth
    costas.process_sample(rotated_symbol, output_symbol);
    float correction_default = costas.get_correction();

    EXPECT_LT(std::abs(correction_after_one), std::abs(correction_default));
}

TEST_F(CostasLoopQPSKTest, ErrorHistoryTracking) {
    cpx input_symbol = create_qpsk_symbol(0);
    cpx rotated_symbol = apply_phase_rotation(input_symbol, M_PI / 6);
    cpx output_symbol;

    // Process a few symbols
    for (int i = 0; i < 5; ++i) {
        costas.process_sample(rotated_symbol, output_symbol);
    }

    auto error_history = costas.get_error_history();

    // Should have non-zero errors recorded
    bool has_nonzero_error = false;
    for (const auto &e : error_history) {
        if (std::abs(e) > 1e-6) {
            has_nonzero_error = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero_error);
}

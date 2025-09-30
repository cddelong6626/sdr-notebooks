#include "sdrlib/carrier_recovery.hpp"
#include "sdrlib/types.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace sdrlib;

TEST(CostasLoopQPSK, BasicFunctionality) {
    float loop_bw = 1.0f / 20.0f;
    carrier_recovery::CostasLoopQPSK costas(loop_bw);

    // Verify initial state
    EXPECT_FLOAT_EQ(costas.get_loop_bw(), loop_bw);
    EXPECT_FLOAT_EQ(costas.get_correction(), 0.0f);

    // Process a known input symbol and check output
    cpx input_symbol = cpx(1.0f, 1.0f); // Example QPSK symbol
    cpx output_symbol;
    costas.process_sample(input_symbol, output_symbol);

    // Since the initial correction is zero, output should match input
    EXPECT_NEAR(output_symbol.real(), input_symbol.real(), 1e-6);
    EXPECT_NEAR(output_symbol.imag(), input_symbol.imag(), 1e-6);

    // Simulate a phase error and process again
    float phase_error = M_PI / 8; // 22.5 degrees
    cpx rotated_symbol = input_symbol * kfr::cexp(cpx(0.0f, phase_error));
    costas.process_sample(rotated_symbol, output_symbol);

    // The output should be closer to the original symbol after correction
    EXPECT_NEAR(output_symbol.real(), input_symbol.real(), 0.5f);
    EXPECT_NEAR(output_symbol.imag(), input_symbol.imag(), 0.5f);

    // Check that error history is being recorded
    // DEBUG: print out error_history
    std::cout << "Error history: ";
    for (const auto &e : costas.get_error_history()) {
        std::cout << e << " ";
    }
    std::cout << std::endl;

    auto error_history = costas.get_error_history();
    EXPECT_GT(error_history[0], 0.0f); // There should be some error recorded

    // Reset the loop and verify state
    costas.reset();
    EXPECT_FLOAT_EQ(costas.get_correction(), 0.0f);
    error_history = costas.get_error_history();
    for (const auto &e : error_history) {
        EXPECT_FLOAT_EQ(e, 0.0f); // Error history should be cleared
    }
}
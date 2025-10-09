#include "sdrlib/modulation.hpp"
#include <complex>
#include <gtest/gtest.h>

using namespace sdrlib;
using namespace modulation;

TEST(ModulationTest, QPSKModulation) {
    const int num_symbols = 4;
    int input_bits[num_symbols] = {0b00, 0b01, 0b10, 0b11};
    sdrlib::cpx expected_symbols[num_symbols] = {
        sdrlib::cpx(+1, +1), // 00
        sdrlib::cpx(+1, -1), // 01
        sdrlib::cpx(-1, +1), // 10
        sdrlib::cpx(-1, -1)  // 11
    };
    sdrlib::cpx modulated_symbols[num_symbols];

    // Modulate
    modulate_qpsk(input_bits, modulated_symbols, num_symbols);

    // Check that modulated symbols match expected
    for (int i = 0; i < num_symbols; ++i) {
        EXPECT_NEAR(modulated_symbols[i].real(), expected_symbols[i].real(), 1e-6)
            << "at index " << i;
        EXPECT_NEAR(modulated_symbols[i].imag(), expected_symbols[i].imag(), 1e-6)
            << "at index " << i;
    }
}

TEST(ModulationTest, QPSKDemodulation) {
    const int num_symbols = 4;
    sdrlib::cpx input_symbols[num_symbols] = {
        sdrlib::cpx(+1, +1), // 00
        sdrlib::cpx(+1, -1), // 01
        sdrlib::cpx(-1, +1), // 10
        sdrlib::cpx(-1, -1)  // 11
    };
    int expected_bits[num_symbols] = {0b00, 0b01, 0b10, 0b11};
    int demodulated_bits[num_symbols];

    // Demodulate
    demodulate_qpsk(input_symbols, demodulated_bits, num_symbols);

    // Check that demodulated bits match expected
    for (int i = 0; i < num_symbols; ++i) {
        EXPECT_EQ(demodulated_bits[i], expected_bits[i]) << "at index " << i;
    }
}

TEST(ModulationTest, QPSKModDemod) {
    const int num_symbols = 8;
    int input_bits[num_symbols] = {0b00, 0b01, 0b10, 0b11, 0b00, 0b01, 0b10, 0b11};
    sdrlib::cpx modulated_symbols[num_symbols];
    int demodulated_bits[num_symbols];

    // Modulate
    modulate_qpsk(input_bits, modulated_symbols, num_symbols);

    // Demodulate
    demodulate_qpsk(modulated_symbols, demodulated_bits, num_symbols);

    // Check that demodulated bits match original input bits
    for (int i = 0; i < num_symbols; ++i) {
        EXPECT_EQ(input_bits[i], demodulated_bits[i]) << "Mismatch at index " << i;
    }
}

TEST(ModulationTest, OptimumDecider) {
    const int num_symbols = 4;
    sdrlib::cpx input_symbols[num_symbols] = {
        sdrlib::cpx(0.9f, 1.1f),   // Close to +1+j -> 00
        sdrlib::cpx(1.2f, -0.8f),  // Close to +1-j -> 01
        sdrlib::cpx(-1.1f, 0.9f),  // Close to -1+j -> 10
        sdrlib::cpx(-0.9f, -1.2f)  // Close to -1-j -> 11
    };
    int expected_bits[num_symbols] = {0b00, 0b01, 0b10, 0b11};
    int decided_bits[num_symbols];

    // Use optimum decision maker
    optimum_decider_qpsk(input_symbols, decided_bits, num_symbols);

    // Check that decided bits match expected
    for (int i = 0; i < num_symbols; ++i) {
        EXPECT_EQ(decided_bits[i], expected_bits[i]) << "at index " << i;
    }
}
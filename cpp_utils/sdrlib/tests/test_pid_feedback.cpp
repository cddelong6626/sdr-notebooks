#include "sdrlib/pid_feedback.hpp"
#include <gtest/gtest.h>

using namespace sdrlib::control;

TEST(PIDFeedbackTest, ZeroGainReturnsZero) {
    PIDFeedback pid(0.0f, 0.0f, 0.0f);
    EXPECT_FLOAT_EQ(pid.process(1.0f), 0.0f);
    EXPECT_FLOAT_EQ(pid.process(-1.0f), 0.0f);
}

TEST(PIDFeedbackTest, ProportionalOnly) {
    PIDFeedback pid(2.0f, 0.0f, 0.0f);
    EXPECT_FLOAT_EQ(pid.process(1.5f), 3.0f);
    EXPECT_FLOAT_EQ(pid.process(-2.0f), -4.0f);
}

TEST(PIDFeedbackTest, IntegralOnly) {
    PIDFeedback pid(0.0f, 1.0f, 0.0f);
    EXPECT_FLOAT_EQ(pid.process(1.0f), 1.0f);
    EXPECT_FLOAT_EQ(pid.process(2.0f), 3.0f);  // sum_e = 1+2=3
    EXPECT_FLOAT_EQ(pid.process(-1.0f), 2.0f); // sum_e = 3+(-1)=2
}

TEST(PIDFeedbackTest, DerivativeOnly) {
    PIDFeedback pid(0.0f, 0.0f, 1.0f);
    EXPECT_FLOAT_EQ(pid.process(1.0f), 1.0f);  // prev_e=0, e=1, diff=1
    EXPECT_FLOAT_EQ(pid.process(4.0f), 3.0f);  // prev_e=1, e=4, diff=3
    EXPECT_FLOAT_EQ(pid.process(2.0f), -2.0f); // prev_e=4, e=2, diff=-2
}

TEST(PIDFeedbackTest, PIDCombined) {
    PIDFeedback pid(1.0f, 0.5f, 0.1f);
    float out1 = pid.process(2.0f);                                           // sum_e=2, prev_e=2
    EXPECT_FLOAT_EQ(out1, 1.0f * 2.0f + 0.5f * 2.0f + 0.1f * 2.0f);          // 2+1+0.2=3.2
    float out2 = pid.process(3.0f);                                           // sum_e=5, prev_e=3
    EXPECT_FLOAT_EQ(out2, 1.0f * 3.0f + 0.5f * 5.0f + 0.1f * (3.0f - 2.0f)); // 3+2.5+0.1=5.6
}

TEST(PIDFeedbackTest, ResetWorks) {
    PIDFeedback pid(0.0f, 1.0f, 0.0f);
    pid.process(2.0f);
    pid.process(3.0f);
    pid.reset();
    EXPECT_FLOAT_EQ(pid.process(1.0f), 1.0f);
}

TEST(PIDFeedbackTest, Multipleprocesss) {
    PIDFeedback pid(0.5f, 0.2f, 0.1f);
    float out1 = pid.process(1.0f);                                             // sum_e=1, prev_e=1
    float out2 = pid.process(2.0f);                                             // sum_e=3, prev_e=2
    float out3 = pid.process(-1.0f);                                            // sum_e=2, prev_e=-1
    EXPECT_FLOAT_EQ(out1, 0.5f * 1.0f + 0.2f * 1.0f + 0.1f * 1.0f);            // 0.5+0.2+0.1=0.8
    EXPECT_FLOAT_EQ(out2, 0.5f * 2.0f + 0.2f * 3.0f + 0.1f * (2.0f - 1.0f));   // 1+0.6+0.1=1.7
    EXPECT_FLOAT_EQ(out3, 0.5f * -1.0f + 0.2f * 2.0f + 0.1f * (-1.0f - 2.0f)); // -0.5+0.4-0.3=-0.4
}
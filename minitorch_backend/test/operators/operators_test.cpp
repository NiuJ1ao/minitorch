#include "gtest/gtest.h"
#include "operators/operators.h"
#include "test_strategies.h"

namespace minitorch{

class OperatorsTest : public testing::TestWithParam<double> {};

// //////////////////
// Basic Tests
// //////////////////

TEST_P(OperatorsTest, SameAsCPPTest) 
{
    double x = GetParam(), y = GetParam();
    ASSERT_DOUBLE_EQ(operators::add(x, y), x + y);
    ASSERT_DOUBLE_EQ(operators::mul(x, y), x * y);
    ASSERT_DOUBLE_EQ(operators::neg(x), -x);
    ASSERT_DOUBLE_EQ(operators::max(x, y), x > y ? x : y);
    if (std::abs(x) > 1e-5) {
        ASSERT_DOUBLE_EQ(operators::inv(x), 1.0 / x);
    }
}

TEST_P(OperatorsTest, ReLUTest) 
{
    double x = GetParam();
    if (x > 0) 
        ASSERT_EQ(operators::relu(x), x);
    else
        ASSERT_EQ(operators::relu(x), 0.0);
}

TEST_P(OperatorsTest, ReLUBackTest) 
{
    double x = GetParam(), y = GetParam();
    if (x > 0)
        ASSERT_EQ(operators::relu_back(x, y), y);
    else
        ASSERT_EQ(operators::relu_back(x, y), 0.0);
}

TEST_P(OperatorsTest, IdTest) 
{
    double x = GetParam();
    ASSERT_EQ(operators::id(x), x);
}

TEST_P(OperatorsTest, ItTest) 
{
    double x = GetParam();
    ASSERT_EQ(operators::lt(x - 1.0, x), 1.0);
    ASSERT_EQ(operators::lt(x, x - 1.0), 0.0);
}

TEST_P(OperatorsTest, MaxTest) 
{
    double x = GetParam();
    ASSERT_EQ(operators::max(x - 1.0, x), x);
    ASSERT_EQ(operators::max(x, x - 1.0), x);
    ASSERT_EQ(operators::max(x + 1.0, x), x + 1.0);
    ASSERT_EQ(operators::max(x, x + 1.0), x + 1.0);
}

TEST_P(OperatorsTest, EqTest) 
{
    double x = GetParam();
    ASSERT_EQ(operators::max(x, x), x);
    ASSERT_EQ(operators::max(x, x - 1.0), x);
    ASSERT_EQ(operators::max(x, x + 1.0), x + 1.0);
}


// /////////////////
// Property Tests
// /////////////////

TEST_P(OperatorsTest, SigmoidTest) 
{
    double x = GetParam();
    ASSERT_GE(operators::sigmoid(x), 0.0);
    ASSERT_LE(operators::sigmoid(x), 1.0);
    ASSERT_NEAR(operators::sigmoid(-x), 1.0 - operators::sigmoid(x), 1e-5);
    ASSERT_DOUBLE_EQ(operators::sigmoid(0.0), 0.5);
    ASSERT_LE(operators::sigmoid(-5.0), operators::sigmoid(5.0));
}

TEST_P(OperatorsTest, SymmetricTest) 
{
    double x = GetParam(), y = GetParam();
    ASSERT_EQ(operators::mul(x, y), operators::mul(y, x));
}


INSTANTIATE_TEST_SUITE_P(
    RandomFloatValues,
    OperatorsTest,
    ::testing::ValuesIn(generate_random_doubles(6))
);
}

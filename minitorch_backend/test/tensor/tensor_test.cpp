#include <random>
#include <stdexcept>

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "tensor/tensor.h"
#include "test_strategies.h"

namespace minitorch {

TEST(TensorLayoutTest, LayoutTest) 
{
    tensor_buffer<double> data(3 * 5);
    std::iota (data.begin(), data.end(), 0.0f);
    tensor_shape shape = {3, 5}, expected_shape = {3, 5};
    tensor_strides strides = {5, 1};
    auto tensor_data = TensorData<double>(data, shape, strides);
    ASSERT_TRUE(tensor_data.is_contiguous());
    tensor_index idx = {1, 0};
    ASSERT_DOUBLE_EQ(tensor_data.index(idx), 5.0f);
    idx = {1, 2};
    ASSERT_DOUBLE_EQ(tensor_data.index(idx), 7.0f);

    shape = {5, 3};
    strides = {1, 5};
    tensor_data = TensorData<double>(data, shape, strides);
    ASSERT_FALSE(tensor_data.is_contiguous());
    ASSERT_THAT(tensor_data.shape, testing::ElementsAre(5, 3));

    data.resize(4 * 2 * 2);
    std::iota(data.begin(), data.end(), 0.0f);
    shape = {4, 2, 2};
    tensor_data = TensorData<double>(data, shape);
    ASSERT_THAT(tensor_data.strides, testing::ElementsAre(4, 2, 1));
}

TEST(TensorTest, ShapeBroadcastTest) 
{
    tensor_shape a = {1,}, b = {5, 5};
    ASSERT_THAT(TensorData<double>::shape_broadcast(a, b), testing::ElementsAre(5, 5));
    
    a = {5, 5}, b = {1,};
    ASSERT_THAT(TensorData<double>::shape_broadcast(a, b), testing::ElementsAre(5, 5));

    a = {1, 5, 5}, b = {5, 5};
    ASSERT_THAT(TensorData<double>::shape_broadcast(a, b), testing::ElementsAre(1, 5, 5));

    a = {5, 1, 5, 1}, b = {1, 5, 1, 5};
    ASSERT_THAT(TensorData<double>::shape_broadcast(a, b), testing::ElementsAre(5, 5, 5, 5));

    a = {5, 7, 5, 1}, b = {1, 5, 1, 5};
    EXPECT_THROW(TensorData<double>::shape_broadcast(a, b), std::runtime_error);

    a = {5, 2}, b = {5,};
    EXPECT_THROW(TensorData<double>::shape_broadcast(a, b), std::runtime_error);

    a = {2, 5}, b = {5,};
    ASSERT_THAT(TensorData<double>::shape_broadcast(a, b), testing::ElementsAre(2, 5));
}


class TensorTest : public testing::TestWithParam<TensorData<double>> {};

TEST_P(TensorTest, EnumerationTest) 
{
    TensorData<double> tensor_data = GetParam();
    auto indices = tensor_data.indices();
    ASSERT_EQ(indices.size(), tensor_data.size);
    for (auto& ind: indices) {
        for (size_t j = 0; j < ind.size(); j++) {
            ASSERT_GE(ind[j], 0);
            ASSERT_LT(ind[j], tensor_data.shape[j]);
        }
    }
}

TEST_P(TensorTest, IndexTest) 
{
    TensorData<double> tensor_data = GetParam();
    auto indices = tensor_data.indices();
    for (auto& ind: indices) {
        double pos = tensor_data.index(ind);
        ASSERT_GE(pos, 0);
        ASSERT_LT(pos, tensor_data.size);
    }

    tensor_index base(tensor_data.dim, 0);
    base[0] = -1;
    EXPECT_THROW(tensor_data.index(base), std::out_of_range);

    if (tensor_data.dim > 1) {
        tensor_index base(tensor_data.dim - 1, 0);
        EXPECT_THROW(tensor_data.index(base), std::length_error);
    }
}

// TEST_P(TensorTest, PermuteTest) 
// {
    
// }

INSTANTIATE_TEST_SUITE_P(
    RandomTensors,
    TensorTest,
    ::testing::ValuesIn(generate_random_tensors(3))
);
}
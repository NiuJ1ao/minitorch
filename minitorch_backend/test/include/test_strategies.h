#include <random>
#include <vector>
#include <algorithm>
#include <cassert>

#include "operators/operators.h"
#include "tensor/tensor.h"

namespace minitorch {

const double MIN_DOUBLE = -100.0f;
const double MAX_DOUBLE = 100.0f;
const int MIN_INT = 1;
const int MAX_INT = 3;
const int MIN_SIZE = 1;
const int MAX_SIZE = 4;

auto random_double(double min, double max) -> double
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(min, max);
    return static_cast<double>(distrib(gen));
}

auto random_int(int min, int max) -> int 
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(min, max);
    return distrib(gen);
}

auto random_shape() -> tensor_shape 
{
    size_t n = random_int(MIN_SIZE, MAX_SIZE); 
    tensor_shape shape(n);
    for (size_t i = 0; i < n; i++) {
        shape[i] = random_int(MIN_INT, MAX_INT);
    }
    return shape;
}

auto random_tensor() -> TensorData<double>
{
    tensor_shape shape = random_shape();
    size_t n = shape.size();
    size_t size = operators::prod(shape);
    tensor_buffer<double> data(size);
    for (size_t i = 0; i < size; i++) {
        data[i] = random_double(MIN_DOUBLE, MAX_DOUBLE);
    }

    tensor_shape permute(n);
    std::iota(permute.begin(), permute.end(), 0);
    std::next_permutation(permute.begin(), permute.end());
    tensor_shape permute_shape(n);
    for (size_t i = 0; i < n; i++) {
        permute_shape[i] = shape[permute[i]];
    }

    std::vector<std::pair<size_t, size_t>> z(n);
    for (size_t i = 0; i < n; i++) {
        z[i] = {i, permute[i]};
    }
    std::sort(z.begin(), z.end(), [](auto &left, auto &right) {
        return left.second < right.second;
    });

    tensor_shape reverse_permute(n);
    for (size_t i = 0; i < n; i++) {
        reverse_permute[i] = z[i].first;
    }

    auto td = TensorData<double>(data, permute_shape);
    auto ret = td.permute(reverse_permute);
    assert(ret.shape[0] == shape[0]);
    return ret;
}

auto generate_random_doubles(size_t n) -> std::vector<double>
{
    std::vector<double> values;
    for (size_t i = 0; i < n; i++) {
        values.push_back(random_double(MIN_DOUBLE, MAX_DOUBLE));
    }
    return values;
}

auto generate_random_tensors(size_t n) -> std::vector<TensorData<double>>
{   
    std::vector<TensorData<double>> values;
    for (size_t i = 0; i < n; i++) {
        values.push_back(random_tensor());
    }
    return values;
}
}
#pragma once

#include <vector>
#include "operators/operators.h"

namespace minitorch {

const size_t MAX_DIM = 32;

template <typename T>
using tensor_buffer = std::vector<T>;
using tensor_index = std::vector<size_t>;
using tensor_shape = std::vector<size_t>;
using tensor_strides = std::vector<size_t>;

template <typename T>
class TensorData {
public:
    TensorData(tensor_buffer<T> &storage, tensor_shape &shape, tensor_strides &stride);
    TensorData(tensor_buffer<T> &storage, tensor_shape &shape);
    virtual ~TensorData() = default;

    static auto broadcast_index(tensor_index &big_index, tensor_shape &big_shape, tensor_shape &shape, tensor_index &out_index);
    static auto to_index(size_t ordinal, tensor_shape &shape, tensor_index &out_index);
    static auto index_to_position(tensor_index &index, const tensor_strides &strides) -> size_t;
    static auto shape_broadcast(tensor_shape &a, tensor_shape &b) -> tensor_shape;

    auto zeros() const -> TensorData<T>;
    auto zeros(tensor_shape &other) const -> TensorData<T>;
    auto to_cuda() const -> void;
    auto is_contiguous() const -> bool;
    auto index(size_t index) const -> size_t {return this->index({index});};
    auto index(tensor_index &index) const -> size_t;
    auto indices() const -> std::vector<tensor_index>;
    auto sample() const -> tensor_index;
    auto get(tensor_index &key) const -> T;
    auto set(tensor_index &key, T val);
    auto tuple() const -> std::tuple<tensor_buffer<T>, tensor_shape, tensor_strides>;
    auto permute(tensor_shape &order) const -> TensorData;

    // A fundamental limitation of pybind11 is that internal conversions 
    // between Python and C++ types involve a copy operation that prevents 
    // pass-by-reference semantics.
    tensor_buffer<T> storage;
    tensor_shape shape;
    tensor_strides strides;
    size_t dim;
    size_t size;

private:
    auto strides_from_shape(tensor_shape &shape) const -> tensor_strides;
};
}
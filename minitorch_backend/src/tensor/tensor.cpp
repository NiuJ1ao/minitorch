#include <vector>
#include <tuple>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cassert>
#include <iostream>

#include "tensor/tensor.h"

namespace minitorch {

    // constructor
    template <typename T>
    TensorData<T>::TensorData(tensor_buffer<T> &storage, tensor_shape &shape, tensor_strides &strides)
    : storage(storage), shape(shape), strides(strides)
    {
        assert(this->shape.size() == this->strides.size());
        dim = this->shape.size();
        size = operators::prod(this->shape);
        assert(this->storage.size() == size);
    }

    template<typename T>
    TensorData<T>::TensorData(tensor_buffer<T> &storage, tensor_shape &shape)
    : storage(storage), shape(shape)
    {
        this->strides = strides_from_shape(shape);
        assert(this->shape.size() == this->strides.size());
        dim = this->shape.size();
        size = operators::prod(this->shape);
        assert(this->storage.size() == size);
    }

    template <typename T>
    auto TensorData<T>::zeros() const -> TensorData<T>
    {
        tensor_buffer<T> storage(size, 0);
        tensor_shape shape(this->shape.begin(), this->shape.end());
        tensor_strides strides(this->strides.begin(), this->strides.end());
        return TensorData<T>(storage, shape, strides);
    }

    template <typename T>
    auto TensorData<T>::zeros(tensor_shape &other) const -> TensorData<T>
    {
        tensor_buffer<T> storage(operators::prod(other), 0);
        tensor_shape shape(other.begin(), other.end());
        return TensorData<T>(storage, shape);
    }

    template <typename T>
    auto TensorData<T>::is_contiguous() const -> bool
    // Check that the layout is contiguous, 
    // i.e. outer dimensions have bigger strides than inner dimensions.
    {
        size_t last = 1e9;
        for (auto &s: strides) {
            if (s > last) return false;
            last = s;
        }
        return true;
    }

    template<typename T>
    auto TensorData<T>::shape_broadcast(tensor_shape &a, tensor_shape &b) -> tensor_shape
    {
        size_t size_a = a.size(), size_b = b.size();

        if (size_a < size_b) return shape_broadcast(b, a);
        
        size_t offset = size_a - size_b;
        tensor_shape padded_b(size_a, 1);
        for (int i = size_b - 1; i >= 0; i--) {
            padded_b[i + offset] = b[i];
        }

        tensor_shape union_shape(size_a);
        for (size_t i = 0; i < size_a; i++) {
            if (a[i] != padded_b[i] && a[i] != 1 && padded_b[i] != 1)
                throw std::runtime_error("shapes cannot be broadcasted.");
            union_shape[i] = std::max(a[i], padded_b[i]);
        }

        return union_shape;
    }

    template<typename T>
    auto TensorData<T>::index(tensor_index &index) const -> size_t
    {   

        if (index.size() != shape.size())
            throw std::length_error("Index must be size of shape.");
        for (size_t i = 0; i < index.size(); i++) {
            if (index[i] >= shape[i])
                throw std::out_of_range("Index out of range.");
            if (index[i] < 0)
                throw std::out_of_range("Negative indexing not supported.");
        }

        // index to position
        return index_to_position(index, strides);
    }

    template<typename T>
    auto TensorData<T>::indices() const -> std::vector<tensor_index>
    {
        tensor_shape lshape(shape.begin(), shape.end());
        std::vector<tensor_index> res(size);
        for (size_t i = 0; i < size; i++) {
            tensor_index out_index(shape.size());
            to_index(i, lshape, out_index);
            res[i] = out_index;
        }
        return res;
    }

    template<typename T>
    auto TensorData<T>::sample() const -> tensor_index
    {
        std::random_device dev;
        std::mt19937 rng(dev());
        tensor_index rand_index(shape.size());
        for (size_t i = 0; i < shape.size(); i++) {
            std::uniform_int_distribution<> distrib(0, shape[i] - 1);
            rand_index[i] = distrib(rng);
        }
        return rand_index;
    }

    template<typename T>
    auto TensorData<T>::get(tensor_index &key) const -> T
    {
        return storage[index(key)];
    }

    template<typename T>
    auto TensorData<T>::set(tensor_index &key, T val)
    {
        storage[index(key)] = val;
    }

    template<typename T>
    auto TensorData<T>::tuple() const -> std::tuple<tensor_buffer<T>, tensor_shape, tensor_strides>
    {
        return std::make_tuple(storage, shape, strides);
    }

    template<typename T>
    auto TensorData<T>::permute(tensor_shape &order) const -> TensorData
    {
        size_t n = order.size();
        tensor_index order_sorted(n);
        std::partial_sort_copy(order.begin(), order.end(), order_sorted.begin(), order_sorted.end());
        for (size_t i = 0; i < shape.size(); i++) {
            if (order_sorted[i] != i)
                throw std::runtime_error("Must given a position to each dimension.");
        }

        tensor_shape new_shape(n);
        tensor_strides new_strides(n);
        for (size_t i = 0; i < n; i++) {
            new_shape[i] = shape[order[i]];
            new_strides[i] = strides[order[i]];
        }
        tensor_buffer<T> new_storage(storage.begin(), storage.end());
        return TensorData(new_storage, new_shape, new_strides);
    }

    template <typename T>
    auto TensorData<T>::strides_from_shape(tensor_shape &shape) const -> tensor_strides
    {
        size_t n = shape.size(), offset = 1;
        tensor_strides strides(n, 1);
        for (size_t i = n - 1; i > 0; i--) {
            strides[i - 1] = shape[i] * offset;
            offset *= shape[i];
        }
        return strides;
    }

    template <typename T>
    auto TensorData<T>::broadcast_index(tensor_index &big_index, tensor_shape &big_shape, tensor_shape &shape, tensor_index &out_index) 
    {
        size_t offset = big_shape.size() - shape.size();
        for (size_t i = 0; i < out_index.size(); i++)
            out_index[i] = shape[i] == 1 ? 0 : big_index[offset + i];
    }

    template<typename T>
    auto TensorData<T>::to_index(size_t ordinal, tensor_shape &shape, tensor_index &out_index)
    {
        if (shape.size() == 1) {
            out_index[0] = ordinal;
            return;
        }
        
        for (size_t i = shape.size() - 1; i > 0; i--) {
            out_index[i] = ordinal % shape[i];
            ordinal /= shape[i];
        }
    }

    template<typename T>
    auto TensorData<T>::index_to_position(tensor_index &index, const tensor_strides &strides) -> size_t
    {
        size_t pos = 0;
        for (size_t i = 0; i < strides.size(); i++)
            pos += index[i] * strides[i];
        return pos;
    }

    template class TensorData<double>;
}

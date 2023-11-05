#include <vector>
#include <stdexcept>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tensor/ndarray.h"

namespace minitorch {

namespace py = pybind11;

template<typename T>
NDArray<T>::NDArray(size_t size) : size_(size)
{
    if (posix_memalign((void**)&this->data, ALIGNMENT, size_ * sizeof(T))) 
        throw std::bad_alloc();
    for (size_t i = 0; i < size_; i++) {
        data[i] = T();
    }
}

template<typename T>
NDArray<T>::NDArray(const size_t size, T val) : size_(size)
{
    if (posix_memalign((void**)&this->data, ALIGNMENT, size_ * sizeof(T))) 
        throw std::bad_alloc();
    for (size_t i = 0; i < size_; i++) {
        data[i] = val;
    }
}

template<typename T>
NDArray<T>::NDArray(const std::vector<T>& vec)
{
    size_ = vec.size();
    if (posix_memalign((void**)&this->data, ALIGNMENT, size_ * sizeof(T))) 
        throw std::bad_alloc();
    for (size_t i = 0; i < size_; i++) {
        data[i] = vec[i];
    }
}

template<typename T>
auto NDArray<T>::operator[](size_t idx) -> T&
{
    return data[idx];
}

template<typename T>
void NDArray<T>::assign_slice(const T val)
{
    for (size_t i = 0; i < size_; i++) {
        data[i] = val;
    }
}

template<typename T>
void NDArray<T>::from_numpy(const py::array_t<T>& input_array)
{
    if (input_array.ndim() != 1 || static_cast<size_t>(input_array.size()) != size_) {
        throw std::runtime_error("Input shape mismatch");
    }
    std::memcpy(data, input_array.data(), size_ * sizeof(T)); 
}

template<typename T>
auto NDArray<T>::to_numpy() -> py::array_t<T>
{
    return py::array_t<T>({size_}, {sizeof(T)}, data);
}


auto broadcast_index(NDArray<int> &big_index, NDArray<int> &big_shape, NDArray<int> &shape, NDArray<int> &out_index) 
{
    size_t offset = big_shape.size() - shape.size();
    for (size_t i = 0; i < out_index.size(); i++)
        out_index[i] = shape[i] == 1 ? 0 : big_index[offset + i];
}

auto to_index(size_t ordinal, NDArray<int>& shape, NDArray<int>& out_index)
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

auto index_to_position(NDArray<int> &index, NDArray<int> &strides) -> size_t
{
    size_t pos = 0;
    for (size_t i = 0; i < strides.size(); i++)
        pos += index[i] * strides[i];
    return pos;
}

auto shape_broadcast(NDArray<int>& a, NDArray<int>& b) -> NDArray<int>
{
    size_t size_a = a.size(), size_b = b.size();

    if (size_a < size_b) return shape_broadcast(b, a);
    
    size_t offset = size_a - size_b;
    NDArray<int> padded_b(size_a, 1);
    for (int i = size_b - 1; i >= 0; i--) {
        padded_b[i + offset] = b[i];
    }

    NDArray<int> union_shape(size_a);
    for (size_t i = 0; i < size_a; i++) {
        if (a[i] != padded_b[i] && a[i] != 1 && padded_b[i] != 1)
            throw std::runtime_error("shapes cannot be broadcasted.");
        union_shape[i] = std::max(a[i], padded_b[i]);
    }

    return union_shape;
}

template class NDArray<int>;
template class NDArray<double>;

template<typename T>
void declare_array(py::module &m, const std::string &typestr)
{
    std::string pyclass_name = std::string("NDArray_") + typestr;

    py::class_<NDArray<T>>(m, pyclass_name.c_str())
        .def(py::init<size_t>())
        .def(py::init<size_t, T>())
        .def(py::init<std::vector<T>>())
        .def("__getitem__", [](NDArray<T> &a, size_t idx) {
            if (idx >= a.size()) throw py::index_error();
            return a[idx];
        })
        .def("__setitem__", [](NDArray<T> &a, size_t idx, const T &value) {
            if (idx >= a.size()) throw py::index_error();
            a[idx] = value;
        })
        .def("__setitem__", [](NDArray<T> &self, py::slice slice, T value) {
            size_t start, stop, step, slicelength;
            if (!slice.compute(self.size(), &start, &stop, &step, &slicelength)) {
                throw py::error_already_set();
            }
            if (step != 1 || slicelength != self.size()) {
                throw std::runtime_error("Only full slices with step 1 are supported");
            }
            self.assign_slice(value);
        })
        .def("__len__", &NDArray<T>::size)
        .def("to_numpy", &NDArray<T>::to_numpy)
        .def("from_numpy", &NDArray<T>::from_numpy);
}

void pybind_ndarray(py::module &m)
{
    declare_array<int>(m, "int32");
    declare_array<double>(m, "float64");

    m.def("broadcast_index", &broadcast_index);
    m.def("to_index", &to_index);
    m.def("index_to_position", &index_to_position);
    m.def("shape_broadcast", &shape_broadcast);
}

} // namespace minitorch







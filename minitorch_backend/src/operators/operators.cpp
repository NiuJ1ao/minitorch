#include <math.h>
#include <vector>
#include <pybind11/pybind11.h>

#include "operators/operators.h"

#define EPS 1e-6

namespace minitorch {

template <typename T>
auto operators::id(T x) -> T
{
    return x;
}

template <typename T>
auto operators::neg(T x) -> T
{
    return -x;
}

template <typename T>
auto operators::log(T x) -> T 
{
    return std::log(x);
}

template <typename T>
auto operators::log_back(T x, T d) -> T
{
    return d / (x + EPS);
}

template <typename T>
auto operators::exp(T x) -> T
{
    return std::exp(x);
}

template <typename T>
auto operators::inv(T x) -> T
{
    return 1.0 / x;
}

template <typename T>
auto operators::inv_back(T x, T d) -> T
{
    return -d / (x * x + EPS);
}

template <typename T>
auto operators::mul(T x, T y) -> T 
{
    return x * y;
}

template <typename T>
auto operators::add(T x, T y) -> T 
{
    return x + y;
}

template <typename T>
auto operators::lt(T x, T y) -> T 
{
    return x < y;
}

template <typename T>
auto operators::eq(T x, T y) -> T 
{
    return x == y;
}

template <typename T>
auto operators::max(T x, T y) -> T
{
    return x > y ? x : y;
}

template <typename T>
auto operators::is_close(T x, T y) -> T
{
    return std::abs(x - y) < 1e-2;
}

template <typename T>
auto operators::sigmoid(T x) -> T
{   
    if (x >= 0)
        return 1.0 / (1.0 + std::exp(-x));
    return std::exp(x) / (1.0 + std::exp(x));
}

template <typename T>
auto operators::sigmoid_back(T x, T d) -> T
{
    return d * x * (1 - x);
}

template <typename T>
auto operators::relu(T x) -> T
{
    return x > 0 ? x : 0.0;
}

template <typename T>
auto operators::relu_back(T x, T d) -> T
{
    return x > 0 ? d : 0.0;
}

template <typename T>
auto operators::prod(std::vector<T> x) -> T 
{
    T res = 1;
    for (T& i: x) res *= i;
    return res;
}


template auto operators::id(double x) -> double;
template auto operators::neg(double x) -> double;
template auto operators::log(double x) -> double;
template auto operators::log_back(double x, double d) -> double;
template auto operators::exp(double x) -> double;
template auto operators::inv(double x) -> double;
template auto operators::inv_back(double x, double d) -> double;
template auto operators::mul(double x, double y) -> double;
template auto operators::add(double x, double y) -> double;
template auto operators::lt(double x, double y) -> double;
template auto operators::eq(double x, double y) -> double;
template auto operators::max(double x, double y) -> double;
template auto operators::is_close(double x, double y) -> double;
template auto operators::sigmoid(double x) -> double;
template auto operators::sigmoid_back(double x, double d) -> double;
template auto operators::relu(double x) -> double;
template auto operators::relu_back(double x, double d) -> double;

template auto operators::prod(std::vector<size_t> x) -> size_t;

namespace py = pybind11;
void pybind_operators(py::module &m)
{
    py::class_<operators>(m, "operators")
        .def("id", &operators::id<double>)
        .def("neg", &operators::neg<double>)
        .def("log", &operators::log<double>)
        .def("log_back", &operators::log_back<double>)
        .def("exp", &operators::exp<double>)
        .def("inv", &operators::inv<double>)
        .def("inv_back", &operators::inv_back<double>)
        .def("mul", &operators::mul<double>)
        .def("add", &operators::add<double>)
        .def("lt", &operators::lt<double>)
        .def("eq", &operators::eq<double>)
        .def("max", &operators::max<double>)
        .def("is_close", &operators::is_close<double>)
        .def("sigmoid", &operators::sigmoid<double>)
        .def("sigmoid_back", &operators::sigmoid_back<double>)
        .def("relu", &operators::relu<double>)
        .def("relu_back", &operators::relu_back<double>);
}

}

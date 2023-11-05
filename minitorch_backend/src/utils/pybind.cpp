#include <pybind11/pybind11.h>

#include "operators/operators.h"
#include "tensor/ndarray.h"

PYBIND11_MODULE(torchbackend, m) {
    using namespace minitorch;
    namespace py = pybind11;

    pybind_ndarray(m);
    pybind_operators(m);
}
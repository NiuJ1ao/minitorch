#include <functional>

#include "tensor/tensor.h"
#include "operators/operators.h"

namespace minitorch {

template <typename T>
using MAP_RET_TYPE = std::function<void(TensorData<T>&, TensorData<T>&)>;
template <typename T>
using MAP_OP_TYPE = std::function<T(T)>;
template <typename T>
using ZIP_RET_TYPE = std::function<void(TensorData<T>&, TensorData<T>&, TensorData<T>&)>;
template <typename T>
using ZIP_OP_TYPE = std::function<T(T, T)>;
template <typename T>
using REDUCE_RET_TYPE = std::function<void(TensorData<T>&, size_t, TensorData<T>&)>;
template <typename T>
using REDUCE_OP_TYPE = std::function<T(T, T)>;
template <typename T>
using MM_RET_TYPE = std::function<void(TensorData<T>&, TensorData<T>&, TensorData<T>&)>;


// #define ZIP_RET_TYPE void (*)(TensorData<T>&, TensorData<T>&, TensorData<T>&)
// #define REDUCE_RET_TYPE void (*)(TensorData<T>&, size_t, TensorData<T>&)

class TensorOps {
public:
    template <typename T>
    static auto map(MAP_OP_TYPE<T> fn) -> MAP_RET_TYPE<T>;

    template <typename T>
    static auto zip(ZIP_OP_TYPE<T> fn) -> ZIP_RET_TYPE<T>;

    template <typename T>
    static auto reduce(REDUCE_OP_TYPE<T> fn, T start = 0) -> REDUCE_RET_TYPE<T>;

    template <typename T>
    static auto matrix_multiply(TensorData<T> a, TensorData<T> b) -> MM_RET_TYPE<T>;

    const static bool cuda = false;
};


class CudaOps : TensorOps {
public:
    template <typename T>
    static auto map(MAP_OP_TYPE<T> fn) -> MAP_RET_TYPE<T>;

    template <typename T>
    static auto zip(ZIP_OP_TYPE<T> fn) -> ZIP_RET_TYPE<T>;

    template <typename T>
    static auto reduce(REDUCE_OP_TYPE<T> fn, T start = 0) -> REDUCE_RET_TYPE<T>;

    template <typename T>
    static auto matrix_multiply(TensorData<T> a, TensorData<T> b) -> MM_RET_TYPE<T>;

    const static bool cuda = true;
};

// template <typename T>
// class TensorBackEnd {
// public:
//     TensorBackEnd(TensorOps<T> ops) : ops_(ops) {};
//     ~TensorBackEnd() = default;

//     TensorData<T> (*neg_map)(TensorData<T>, TensorData<T>) = ops_.map(operators::neg);
//     TensorData<T> (*sigmoid_map)(TensorData<T>, TensorData<T>) = ops_.map(operators::sigmoid);
//     TensorData<T> (*relu_map)(TensorData<T>, TensorData<T>) = ops_.map(operators::relu);
//     TensorData<T> (*log_map)(TensorData<T>, TensorData<T>) = ops_.map(operators::log);
//     TensorData<T> (*exp_map)(TensorData<T>, TensorData<T>) = ops_.map(operators::exp);
//     TensorData<T> (*id_map)(TensorData<T>, TensorData<T>) = ops_.map(operators::id);
//     TensorData<T> (*inv_map)(TensorData<T>, TensorData<T>) = ops_.map(operators::inv);

//     TensorData<T> (*add_zip)(TensorData<T>, TensorData<T>) = ops_.zip(operators::add);
//     TensorData<T> (*mul_zip)(TensorData<T>, TensorData<T>) = ops_.zip(operators::mul);
//     TensorData<T> (*lt_zip)(TensorData<T>, TensorData<T>) = ops_.zip(operators::lt);
//     TensorData<T> (*eq_zip)(TensorData<T>, TensorData<T>) = ops_.zip(operators::eq);
//     TensorData<T> (*is_close_zip)(TensorData<T>, TensorData<T>) = ops_.zip(operators::is_close);
//     TensorData<T> (*relu_back_zip)(TensorData<T>, TensorData<T>) = ops_.zip(operators::relu_back);
//     TensorData<T> (*log_back_zip)(TensorData<T>, TensorData<T>) = ops_.zip(operators::log_back);
//     TensorData<T> (*inv_back_zip)(TensorData<T>, TensorData<T>) = ops_.zip(operators::inv_back);
//     TensorData<T> (*sigmoid_back_zip)(TensorData<T>, TensorData<T>) = ops_.zip(operators::sigmoid_back);

//     TensorData<T> (*add_reduce)(TensorData<T>, size_t) = ops_.reduce(operators::add, 0.0);
//     TensorData<T> (*mul_reduce)(TensorData<T>, size_t) = ops_.reduce(operators::mul, 1.0);
//     std::function<TensorData<T>(TensorData<T>, TensorData<T>)> matrix_multiply = ops_.matrix_multiply;
//     bool cuda = ops_.cuda;

// private:
//     TensorOps<T> ops_;
// };

// template <typename T>
// class CudaOps : TensorOps<T> {
// public:
//     static auto map(T (*func)(T)) -> MAP_RET_TYPE;
//     static auto zip(T (*func)(T, T)) -> ZIP_RET_TYPE;
//     static auto reduce(T (*func)(T, T), T start = 0) -> REDUCE_RET_TYPE;
//     static void matrix_multiply(TensorData<T> &a, TensorData<T> &b, TensorData<T> &c);
//     const static bool cuda = true;
// };
}

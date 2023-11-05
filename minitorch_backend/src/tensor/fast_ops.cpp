#include <functional>
#include <omp.h>

#include "tensor/tensor_ops.h"
#include "tensor/tensor.h"
#include "operators/operators.h"


namespace minitorch {

template<typename T>
auto TensorOps::map(MAP_OP_TYPE<T> fn) -> MAP_RET_TYPE<T>
{
    // return [&](TensorData<T> &a, TensorData<T> &out) -> void
    // {
    //     #pragma omp parallel
    //     {
    //         size_t i;
    //         #pragma omp for
    //         {
    //             for (i = 0; i < out.size; i++) {
    //                 tensor_index out_index(MAX_DIM), in_index(MAX_DIM);
    //                 TensorData::to_index(i, out.shape, out_index);
    //                 TensorData::broadcast_index(out_index, out.shape, in.shape, in_index);
    //                 out[TensorData::index_to_position(out_index, out.strides)] = fn(
    //                     in_storage[TensorData::index_to_position(in_index, in.strides)]
    //                 );
    //             }
    //         }
    //     }
    // };
}

template<typename T>
auto TensorOps::zip(ZIP_OP_TYPE<T> fn) -> ZIP_RET_TYPE<T>
{
    
}

}
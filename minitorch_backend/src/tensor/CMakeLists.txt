add_library(
    minitorch_tensor
    OBJECT
    tensor.cpp
    ndarray.cpp
    fast_ops.cpp
    cuda_ops.cpp
)

set(ALL_OBJECT_FILES
  ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:minitorch_tensor>
  PARENT_SCOPE)
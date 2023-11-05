add_library(
    minitorch_utils
    OBJECT
    pybind.cpp
)

set(ALL_OBJECT_FILES
  ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:minitorch_utils>
  PARENT_SCOPE)
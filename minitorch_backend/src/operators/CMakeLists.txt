add_library(
    minitorch_operators
    OBJECT
    operators.cpp
)

set(ALL_OBJECT_FILES
  ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:minitorch_operators>
  PARENT_SCOPE)

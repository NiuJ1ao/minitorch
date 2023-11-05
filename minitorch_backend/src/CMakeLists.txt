add_subdirectory(operators)
add_subdirectory(tensor)
add_subdirectory(utils)

add_library(torchbackend SHARED ${ALL_OBJECT_FILES})

set(MINITORCH_LIBS
        minitorch_operators
        minitorch_tensor
        minitorch_utils
        )

target_link_libraries(
        torchbackend
        ${MINITORCH_LIBS}
        ${PYTHON_LIBRARIES}
        )

target_include_directories(
        torchbackend PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        ${PYTHON_INCLUDE_DIRS}
        )

pybind11_extension(torchbackend)
pybind11_strip(torchbackend)

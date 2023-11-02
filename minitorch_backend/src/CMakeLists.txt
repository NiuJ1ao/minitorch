add_subdirectory(operators)
add_subdirectory(tensor)

add_library(torchbackend SHARED ${ALL_OBJECT_FILES})

set(MINITORCH_LIBS
        minitorch_operators
        minitorch_tensor
        )

target_link_libraries(
        torchbackend
        ${MINITORCH_LIBS})

target_include_directories(
        torchbackend PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

pybind11_extension(torchbackend)
pybind11_strip(torchbackend)

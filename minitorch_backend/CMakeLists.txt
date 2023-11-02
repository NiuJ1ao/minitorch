cmake_minimum_required(VERSION 3.10)

project(torchbackend C CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        message(STATUS "Setting build type to `Debug` as none was specified.")
        set(CMAKE_BUILD_TYPE "Debug")
endif()

if(EMSCRIPTEN)
        add_compile_options(-fexceptions)
        add_link_options(-fexceptions)
endif()

# People keep running CMake in the wrong folder, completely nuking their project or creating weird bugs.
# This checks if you're running CMake from a folder that already has CMakeLists.txt.
# Importantly, this catches the common case of running it from the root directory.
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" PATH_TO_CMAKELISTS_TXT)

if(EXISTS "${PATH_TO_CMAKELISTS_TXT}")
        message(FATAL_ERROR "Run CMake from a build subdirectory! \"mkdir build ; cd build ; cmake ..\" \
    Some junk files were created in this folder (CMakeCache.txt, CMakeFiles); you should delete those.")
endif()

# #####################################################################################################################
# DEPENDENCIES
# #####################################################################################################################

# Python
find_package(Python COMPONENTS Interpreter Development)

# CTest
enable_testing()

# #####################################################################################################################
# COMPILER SETUP
# #####################################################################################################################
if(NOT DEFINED MINITORCH_SANITIZER)
        set(MINITORCH_SANITIZER address)
endif()

message("Build mode: ${CMAKE_BUILD_TYPE}")
message("${MINITORCH_SANITIZER} sanitizer will be enabled in debug mode.")

# Compiler flags.
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Werror")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-unused-parameter -Wno-attributes") # TODO: remove
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -ggdb -fomit-frame-pointer -foptimize-sibling-calls -fopenmp")
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=${MINITORCH_SANITIZER}") # TODO: Does not work with pybind?
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS}")
message(STATUS "CMAKE_SHARED_LINKER_FLAGS: ${CMAKE_SHARED_LINKER_FLAGS}")

# Includes.
set(MINITORCH_SRC_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/src/include)
set(MINITORCH_TEST_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/test/include)
set(MINITORCH_THIRD_PARTY_INCLUDE_DIR
        ${PROJECT_SOURCE_DIR}/third_party
        ${PROJECT_SOURCE_DIR}/third_party/backward-cpp/include
        ${PROJECT_SOURCE_DIR}/third_party/pybind11/include
)

include_directories(${MINITORCH_SRC_INCLUDE_DIR} ${MINITORCH_TEST_INCLUDE_DIR} ${MINITORCH_THIRD_PARTY_INCLUDE_DIR})
include_directories(BEFORE src) # This is needed for gtest.

message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
include_directories(${Python_INCLUDE_DIRS}) # This is needed for pybind11

function(disable_target_warnings NAME)
        target_compile_options(${NAME} PRIVATE "-w")
endfunction()

# #####################################################################################################################
# Other CMake modules
# MUST BE ADDED AFTER CONFIGURING COMPILER PARAMETERS
# #####################################################################################################################
add_subdirectory(third_party)
add_subdirectory(src)
add_subdirectory(test)

# #####################################################################################################################
# MAKE TARGETS
# #####################################################################################################################

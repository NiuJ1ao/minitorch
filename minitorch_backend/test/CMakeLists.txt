cmake_minimum_required(VERSION 3.10)

include(GoogleTest)

file(GLOB_RECURSE MINITORCH_TEST_SOURCES "${PROJECT_SOURCE_DIR}/test/*/*test.cpp")

# #####################################################################################################################
# MAKE TARGETS
# #####################################################################################################################

# #########################################
# "make check-tests"
# #########################################
add_custom_target(build-tests COMMAND ${CMAKE_CTEST_COMMAND} --show-only)
add_custom_target(check-tests COMMAND ${CMAKE_CTEST_COMMAND} --verbose)

# #########################################
# "make XYZ_test"
# #########################################
foreach (minitorch_test_source ${MINITORCH_TEST_SOURCES})
    # Create a human readable name.
    get_filename_component(minitorch_test_filename ${minitorch_test_source} NAME)
    string(REPLACE ".cpp" "" minitorch_test_name ${minitorch_test_filename})

    # Add the test target separately and as part of "make check-tests".
    add_executable(${minitorch_test_name} EXCLUDE_FROM_ALL ${minitorch_test_source} "${PROJECT_SOURCE_DIR}/third_party/backtrace.cpp")
    add_backward(${minitorch_test_name})
    add_dependencies(build-tests ${minitorch_test_name})
    add_dependencies(check-tests ${minitorch_test_name})

    gtest_discover_tests(${minitorch_test_name}
            EXTRA_ARGS
            --gtest_color=auto
            --gtest_output=xml:${CMAKE_BINARY_DIR}/test/${minitorch_test_name}.xml
            --gtest_catch_exceptions=0
            DISCOVERY_TIMEOUT 120
            PROPERTIES
            TIMEOUT 120
            )

    target_link_libraries(${minitorch_test_name} torchbackend gtest gmock_main)

    # Set test target properties and dependencies.
    set_target_properties(${minitorch_test_name}
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
            COMMAND ${minitorch_test_name}
            )

endforeach()
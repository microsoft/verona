if(NOT DEFINED SOURCE_DIR OR NOT DEFINED CXX_COMPILER)
  message(FATAL_ERROR "This test requires SOURCE_DIR and CXX_COMPILER.")
endif()

set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/build-type-required")
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    -S "${SOURCE_DIR}"
    -B "${build_dir}"
    -G Ninja
    "-DCMAKE_BUILD_TYPE:STRING="
    "-DCMAKE_CXX_COMPILER:FILEPATH=${CXX_COMPILER}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error
  TIMEOUT 60)
file(REMOVE_RECURSE "${build_dir}")

if(result STREQUAL "0")
  message(FATAL_ERROR "Configuration without a build type unexpectedly succeeded.")
endif()
if(NOT "${output}\n${error}" MATCHES "A build type is required")
  message(FATAL_ERROR
    "Configuration failed for an unexpected reason:\n${output}\n${error}")
endif()

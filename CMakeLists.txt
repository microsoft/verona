cmake_minimum_required(VERSION 3.10.0)

# Bootstrap vcpkg before we declare the project
include(cmake/vcpkg.cmake)

project(verona CXX)

message(STATUS "Using toolchain file ${CMAKE_TOOLCHAIN_FILE}")

set (VERONA_DONT_PASS)
macro (set_cache_top VAR HELP TYPE DEFAULT)
  set (${VAR} ${DEFAULT} CACHE ${TYPE} ${HELP})
  list (APPEND VERONA_DONT_PASS ${VAR})
endmacro()
macro (option_top VAR HELP DEFAULT)
  set_cache_top(${VAR} ${HELP} BOOL ${DEFAULT})
endmacro ()

# Other Verona options
option(ENABLE_ASSERTS "Enable asserts even in release builds" OFF)
option_top(RT_TESTS "Including unit tests for the runtime" OFF)
option_top(VERONA_EXPENSIVE_SYSTEMATIC_TESTING "Increase the range of seeds covered by systematic testing" OFF)
option(USE_SCHED_STATS "Track scheduler stats" OFF)
option(USE_ASAN "Use address sanitizer" OFF)
option(VERONA_CI_BUILD "Disable features not sensible for CI" OFF)
option(USE_SYSTEMATIC_TESTING "Enable systematic testing in the runtime" OFF)
option(USE_CRASH_LOGGING "Enable crash logging in the runtime" OFF)
if (NOT MSVC)
  option(CMAKE_EXPORT_COMPILE_COMMANDS "Export compilation commands" ON)
endif ()

##########################################################
# Pass parameters to subbuild.
##########################################################

## Clang-format target
include(cmake/clangformat.cmake)
clangformat_targets()

## Check copyright/license target
include(cmake/copyright.cmake)
copyright_targets()

set(DEFAULT_BUILD_TYPE "Release")
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
endif()

##########################################################
#  Build Verona subproject
##########################################################
SET(VERONA_LOCAL_DIST ${CMAKE_BINARY_DIR}/dist)

add_subdirectory(src)

######################################################
#  Add testing at top level
######################################################
enable_testing()

include(cmake/enable-asserts.cmake)
if (ENABLE_ASSERTS)
  enable_asserts()
endif()

include(ProcessorCount)
ProcessorCount(N)

add_subdirectory(testsuite)

# Adds a target check that runs the tests.
add_custom_target(check)
add_custom_command(TARGET check 
  COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> -j ${N} --output-on-failure --timeout 400 --interactive-debug-mode 0
  USES_TERMINAL
)
add_dependencies(check verona-parser)
add_dependencies(check verona-interop)
add_dependencies(check verona-mlir)

# Main target does not test runtime, build subdirectory for that
# and add to check target.
if (RT_TESTS)
  add_dependencies(check rt_tests)
endif()

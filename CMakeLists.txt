cmake_minimum_required(VERSION 3.10.0)
project(verona-lang CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_STANDARD 17)

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/dist CACHE PATH "Default to installing inside build dir" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

enable_testing()

if(MSVC)
  add_compile_options(/permissive-)
  add_compile_options(/utf-8)
  add_compile_options(/wd4307)
endif()

add_subdirectory(external/CLI11)
add_subdirectory(external/fmt)
add_subdirectory(external/pegmatite)

add_library(cpp-peglib INTERFACE)
target_include_directories(cpp-peglib INTERFACE external/cpp-peglib)

add_subdirectory(src)
add_subdirectory(testsuite)

# A target to run the ctests
# TODO: Add run-time libraries, too, and a check-all for both
include(ProcessorCount)
ProcessorCount(N)
if (NOT N EQUAL 0)
  set(PARALLEL_ARGS --parallel ${N})
endif()

# The `install` target is not supported by add_dependencies
# and doesn't work with msbuild.  This adds a custom target that
# pushes the original target into cmake, so that tests can
# depend on it.
add_custom_target(install_for_test
                  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} ${PARALLEL_ARGS} --config $<CONFIG> --target install
                  COMMENT "Installs distribution."
                  USES_TERMINAL COMMAND_EXPAND_LISTS)

set(CTEST "ctest")
add_custom_target("check"
                  COMMAND ${CTEST} ${PARALLEL_ARGS} -C $<CONFIG>
                  COMMENT "Run ctest tests"
                  USES_TERMINAL COMMAND_EXPAND_LISTS)

add_dependencies("check" "install_for_test")

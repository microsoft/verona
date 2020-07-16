cmake_minimum_required(VERSION 3.10.0)

option (VERONA_LLVM_LOCATION "Location of LLVM build" "")
option (VERONA_ONLY_BUILD_LLVM "Used to build a cache for LLVM build" Off)

set (VERONA_EXTRA_CMAKE_ARGS)

macro(verona_option variable message default)
  option(variable message default)
  list (APPEND VERONA_EXTRA_CMAKE_ARGS
    "-D${variable}=${${variable}}"
  )
endmacro()

verona_option(RT_TESTS "Including unit tests for the runtime" OFF)
verona_option(USE_SCHED_STATS "Track scheduler stats" OFF)
verona_option(USE_ASAN "Use address sanitizer" OFF)
verona_option(VERONA_CI_BUILD "Disable features not sensible for CI" OFF)
verona_option(USE_SYSTEMATIC_TESTING "Enable systematic testing in the runtime" OFF)
verona_option(VERONA_EXPENSIVE_SYSTEMATIC_TESTING "Increase the range of seeds covered by systematic testing" OFF)
verona_option(USE_CRASH_LOGGING "Enable crash logging in the runtime" OFF)

# This is to trick the CMake into building LLVM before Verona.
if (VERONA_LLVM_LOCATION STREQUAL "")
  project(verona CXX)

  set (EXTRA_CMAKE_ARGS)
  list (APPEND EXTRA_CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/mlir_install
    -DLLVM_ENABLE_PROJECTS=mlir
    -DLLVM_TARGETS_TO_BUILD=X86
    -DLLVM_ENABLE_ASSERTIONS=On
    -DLLVM_ENABLE_RTTI=ON
    -DLLVM_ENABLE_EH=ON
  )

  if (MSVC)
  
  else()
  list (APPEND EXTRA_CMAKE_ARGS
    -DLLVM_ENABLE_LLD=ON
  )
  endif()
  
  # Choose sanitizers based on stuff.
  # -DLLVM_USE_SANITIZER=$(Sanitizer) 


  include(ExternalProject)
  ExternalProject_Add(mlir SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/llvm-project/llvm
    CMAKE_ARGS ${EXTRA_CMAKE_ARGS}
    BUILD_ALWAYS true
    USES_TERMINAL_BUILD true
    USES_TERMINAL_CONFIGURE true
  )

  list (APPEND VERONA_EXTRA_CMAKE_ARGS
    -DVERONA_LLVM_LOCATION=${CMAKE_BINARY_DIR}/mlir_install
  )

  if (NOT VERONA_ONLY_BUILD_LLVM)
    ExternalProject_Add(verona SOURCE_DIR ${CMAKE_SOURCE_DIR}
      DEPENDS mlir
      TEST_AFTER_INSTALL true
      BUILD_ALWAYS true
      USES_TERMINAL_BUILD true
      USES_TERMINAL_CONFIGURE true
      USES_TERMINAL_INSTALL true
      USES_TERMINAL_TEST true
      CMAKE_ARGS ${VERONA_EXTRA_CMAKE_ARGS})
  endif ()
  return ()
endif()

project(verona-lang CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_STANDARD 17)

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/dist CACHE PATH "Default to installing inside build dir" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

enable_testing()

# Copied from LLVM, handling release builds with assertions isn't trivial as
# CMake defines -DNDEBUG automatically and we have to clear it.
if( ENABLE_ASSERTS )
  # MSVC doesn't like _DEBUG on release builds.
  if( NOT MSVC )
    message(STATUS "Enabling asserts")
    add_definitions( -D_DEBUG )
  endif()
  # On non-Debug builds cmake automatically defines NDEBUG, so we
  # explicitly undefine it:
  if( NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG" )
    message(STATUS "Removing NDEBUG from existing flags to enable asserts")
    # NOTE: use `add_compile_options` rather than `add_definitions` since
    # `add_definitions` does not support generator expressions.
    add_compile_options($<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-UNDEBUG>)

    # Also remove /D NDEBUG to avoid MSVC warnings about conflicting defines.
    foreach (flags_var_to_scrub
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS_MINSIZEREL
        CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_RELWITHDEBINFO
        CMAKE_C_FLAGS_MINSIZEREL)
      string (REGEX REPLACE "(^| )[/-]D *NDEBUG($| )" " "
        "${flags_var_to_scrub}" "${${flags_var_to_scrub}}")
    endforeach()
  endif()
endif()

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
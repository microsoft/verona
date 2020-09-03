cmake_minimum_required(VERSION 3.10.0)

# CMAKE_HOME_DIRECTORY should not be passed down to another project.
set (VERONA_DONT_PASS CMAKE_HOME_DIRECTORY)
macro (option_top VAR HELP DEFAULT)
  option (${VAR} ${HELP} ${DEFAULT})
  list (APPEND VERONA_DONT_PASS ${VAR})
endmacro ()

option_top (VERONA_DOWNLOAD_LLVM "Download cached version of LLVM" ON)
option_top (VERBOSE_LLVM_DOWNLOAD "Verbose LLVM/MLIR download step" OFF)

message (STATUS "Download LLVM: ${VERONA_DOWNLOAD_LLVM}")
# Lifted from snmalloc. Hard to include with external projects, so copied
macro(clangformat_targets)
  # The clang-format tool is installed under a variety of different names.  Try
  # to find a sensible one.  Only look for versions 9 explicitly - we don't
  # know whether our clang-format file will work with newer versions of the
  # tool.  It does not work with older versions as AfterCaseLabel is not supported
  # in earlier versions.
  #
  # This can always be overriden with `-DCLANG_FORMAT=/path/to/clang-format` if
  # need be.
  find_program(CLANG_FORMAT NAMES
    clang-format-9)

  # If we've found a clang-format tool, generate a target for it, otherwise emit
  # a warning.
  if (${CLANG_FORMAT} STREQUAL "CLANG_FORMAT-NOTFOUND")
    message(WARNING "Not generating clangformat target, no clang-format tool found")
  else ()
    message(STATUS "Generating clangformat target using ${CLANG_FORMAT}")
    file(GLOB_RECURSE ALL_SOURCE_FILES src/*.cc src/*.h src/*.hh)
    add_custom_target(
      clangformat
      COMMAND ${CLANG_FORMAT}
      -i
      ${ALL_SOURCE_FILES})
  endif()
endmacro()

clangformat_targets()

option(ENABLE_ASSERTS "Enable asserts even in release builds" OFF)
option(RT_TESTS "Including unit tests for the runtime" OFF)
option(USE_SCHED_STATS "Track scheduler stats" OFF)
option(USE_ASAN "Use address sanitizer" OFF)
option(VERONA_CI_BUILD "Disable features not sensible for CI" OFF)
option(USE_SYSTEMATIC_TESTING "Enable systematic testing in the runtime" OFF)
option(VERONA_EXPENSIVE_SYSTEMATIC_TESTING "Increase the range of seeds covered by systematic testing" OFF)
option(USE_CRASH_LOGGING "Enable crash logging in the runtime" OFF)
if (NOT MSVC)
  option(CMAKE_EXPORT_COMPILE_COMMANDS "Export compilation commands" ON)
endif ()


# This is to trick the CMake into building LLVM before Verona.
# We use two External Projects, so that the LLVM build can complete and install
# before we start the Verona one.  This is required as we consume CMake
# artifacts, which are not available otherwise.
include(ExternalProject)
if (NOT DEFINED VERONA_LLVM_LOCATION)
  project(verona CXX)
  set(DEFAULT_BUILD_TYPE "Release")
  if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
  endif()

  set (MLIR_INSTALL ${CMAKE_BINARY_DIR}/$<CONFIG>/mlir)

  set (EXTERNAL_EXTRA_CMAKE_ARGS)
  list (APPEND EXTERNAL_EXTRA_CMAKE_ARGS 
    -DVERONA_DOWNLOAD_LLVM=${VERONA_DOWNLOAD_LLVM}
    -DMLIR_INSTALL=${MLIR_INSTALL}
    -DCMAKE_BUILD_TYPE=$<CONFIG>)

  if (VERONA_DOWNLOAD_LLVM AND DEFINED LLVM_EXTRA_CMAKE_ARGS)
    message (WARNING "Ignoring LLVM_EXTRA_CMAKE_ARGS as cached LLVM.")
  endif ()

  if (NOT VERONA_DOWNLOAD_LLVM)
    if (NOT DEFINED LLVM_EXTRA_CMAKE_ARGS)
      set (LLVM_EXTRA_CMAKE_ARGS)
    endif ()

    list (APPEND EXTERNAL_EXTRA_CMAKE_ARGS
      -DLLVM_EXTRA_CMAKE_ARGS=${LLVM_EXTRA_CMAKE_ARGS}
    )
  else ()
    list (APPEND EXTERNAL_EXTRA_CMAKE_ARGS
      -DVERBOSE_LLVM_DOWNLOAD=${VERBOSE_LLVM_DOWNLOAD}
    )
  endif ()


  ExternalProject_Add(external
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/external
    CMAKE_ARGS ${EXTERNAL_EXTRA_CMAKE_ARGS} 
    BUILD_ALWAYS true
    INSTALL_COMMAND ""
    TEST_COMMAND ""
    USES_TERMINAL_BUILD true
    USES_TERMINAL_CONFIGURE true
  )

  if (DEFINED VERONA_EXTRA_CMAKE_ARGS)
    message (WARNING "Do not use VERONA_EXTRA_CMAKE_ARGS! Value ignored")
  endif ()
  set (VERONA_EXTRA_CMAKE_ARGS)

  # Point Verona at the MLIR
  list (APPEND VERONA_EXTRA_CMAKE_ARGS
    -DVERONA_LLVM_LOCATION=${MLIR_INSTALL}/install
  )

  # Define project level defaults
  list (APPEND VERONA_EXTRA_CMAKE_ARGS
    -DCMAKE_CXX_STANDARD=17)

  # Use top-level install directory when building
  # as a subproject.
  list (APPEND VERONA_EXTRA_CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/dist)
  list (APPEND VERONA_DONT_PASS CMAKE_INSTALL_PREFIX)

  get_cmake_property(CACHE_VARS CACHE_VARIABLES)
  foreach (CACHE_VAR ${CACHE_VARS})
    list (FIND VERONA_DONT_PASS ${CACHE_VAR} INDEX)
    if (${INDEX} EQUAL -1)
      list(APPEND VERONA_EXTRA_CMAKE_ARGS
        -D${CACHE_VAR}=${${CACHE_VAR}}
      )
    endif()
  endforeach ()

  # Current the ExternalProject_Add_StepTargets seems to be generating something
  # that is breaking CI for MSVC. So on MSVC with have a single build and test
  # target. On other platforms we keep them separate.
  if (MSVC)
    set (SEPARATE_TEST_TARGET false)
  else ()
    set (SEPARATE_TEST_TARGET true)
  endif()

  # We use ctest directly as test command, pass it -j
  include(ProcessorCount)
  ProcessorCount(N)

  ExternalProject_Add(verona
    SOURCE_DIR ${CMAKE_SOURCE_DIR}
    DEPENDS external
    TEST_COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> -j ${N} --timeout 400 --output-on-failure --interactive-debug-mode 0
    TEST_AFTER_INSTALL true
    TEST_EXCLUDE_FROM_MAIN ${SEPARATE_TEST_TARGET}
    BUILD_ALWAYS true
    USES_TERMINAL_BUILD true
    USES_TERMINAL_CONFIGURE true
    USES_TERMINAL_INSTALL true
    USES_TERMINAL_TEST true
    CMAKE_ARGS ${VERONA_EXTRA_CMAKE_ARGS})

  if (${SEPARATE_TEST_TARGET})
    ExternalProject_Add_StepTargets(verona test)

    add_custom_target(check DEPENDS verona-test)
  endif ()
  return ()
endif()

project(verona-lang CXX)

message(STATUS "Build Type for Verona ${CMAKE_BUILD_TYPE}")
set(CMAKE_CONFIGURATION_TYPES  Release Debug RelWithDebInfo)
message(STATUS "Build types ${CMAKE_CONFIGURATION_TYPES}")

if (VERONA_CI_BUILD)
  # Specify policy to go into child projects.
  set (CMAKE_POLICY_DEFAULT_CMP0077 NEW)
  set (SNMALLOC_CI_BUILD ON)
endif ()

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

add_subdirectory(external/CLI11 EXCLUDE_FROM_ALL)
add_subdirectory(external/fmt EXCLUDE_FROM_ALL)
add_subdirectory(external/pegmatite EXCLUDE_FROM_ALL)

add_library(cpp-peglib INTERFACE)
target_include_directories(cpp-peglib INTERFACE external/cpp-peglib)

add_subdirectory(src)
add_subdirectory(testsuite)

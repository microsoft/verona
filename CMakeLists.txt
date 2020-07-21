cmake_minimum_required(VERSION 3.10.0)

option (VERONA_DOWNLOAD_LLVM "Download cached version of LLVM" ON)

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

# This is used to create options that are passed through to the main Verona
# build
if (NOT DEFINED VERONA_EXTRA_CMAKE_ARGS)
  set (VERONA_EXTRA_CMAKE_ARGS)
endif ()
macro(verona_option variable message default)
  option(variable message default)
  list (APPEND VERONA_EXTRA_CMAKE_ARGS
    "-D${variable}=${${variable}}"
  )
endmacro()

verona_option(ENABLE_ASSERTS "Enable asserts even in release builds" OFF)
verona_option(RT_TESTS "Including unit tests for the runtime" OFF)
verona_option(USE_SCHED_STATS "Track scheduler stats" OFF)
verona_option(USE_ASAN "Use address sanitizer" OFF)
verona_option(VERONA_CI_BUILD "Disable features not sensible for CI" OFF)
verona_option(USE_SYSTEMATIC_TESTING "Enable systematic testing in the runtime" OFF)
verona_option(VERONA_EXPENSIVE_SYSTEMATIC_TESTING "Increase the range of seeds covered by systematic testing" OFF)
verona_option(USE_CRASH_LOGGING "Enable crash logging in the runtime" OFF)

# This is to trick the CMake into building LLVM before Verona.
# We use two External Projects, so that the LLVM build can complete and install
# before we start the Verona one.  This is required as we consume CMake
# artifacts, which are not available otherwise.
include(ExternalProject)
if (NOT DEFINED VERONA_LLVM_LOCATION)
  project(verona CXX)
  # Restrict types of builds to ones where we have cached LLVM
  set(default_build_type "Release")
  if (MSVC)
    if (NOT CMAKE_CONFIGURATION_TYPES)
      if (NOT CMAKE_BUILD_TYPE)
        set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDbgInfo" CACHE STRING "Choose the types of build." FORCE)
      else()
        set(CMAKE_CONFIGURATION_TYPES "${CMAKE_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
      endif()
      list (APPEND VERONA_EXTRA_CMAKE_ARGS
        -DCMAKE_CONFIGURATION_TYPES=${CMAKE_CONFIGURATION_TYPES})
    endif ()
  else ()
    if (NOT CMAKE_BUILD_TYPE)
      message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
      set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
    endif()
    if (NOT CMAKE_CONFIGURATION_TYPES)
      set(CMAKE_CONFIGURATION_TYPES ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the types of build." FORCE)
    endif ()
  endif ()

  if (VERONA_DOWNLOAD_LLVM)
    find_package(Git)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-parse --short=11 HEAD
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external/llvm-project"
      OUTPUT_VARIABLE LLVM_PACKAGE_GIT_VERSION
    )

    string(TOLOWER ${CMAKE_SYSTEM_NAME} PLATFORM)

    set(MLIR_TARGETS)
    # Need loop to download all types for possible configurations
    # Required by multi-build systems like msbuild
    foreach (BUILD_TYPE_MIXED ${CMAKE_CONFIGURATION_TYPES})
      string (TOLOWER ${BUILD_TYPE_MIXED} BUILD_TYPE)
      if (${BUILD_TYPE} STREQUAL relwithdebinfo)
        # RelWithDebInfo should use release LLVM.
        set(LLVM_BUILD_TYPE release)
      elseif (WIN32)
        set(LLVM_BUILD_TYPE ${BUILD_TYPE})
      else ()
        # Use release everywhere it works.
        set(LLVM_BUILD_TYPE release)
      endif ()

      set(PKG_NAME verona-llvm-install-x86_64-${PLATFORM}-${LLVM_BUILD_TYPE}-${LLVM_PACKAGE_GIT_VERSION})

      ExternalProject_Add(mlir-${BUILD_TYPE}
        URL https://verona.blob.core.windows.net/llvmbuild/${PKG_NAME}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        TEST_COMMAND ""
        USES_TERMINAL_DOWNLOAD true
      )
      message(STATUS "Down task for ${BUILD_TYPE}")
      list (APPEND MLIR_TARGETS mlir-${BUILD_TYPE})
    endforeach()

    set (MLIR_TYPE mlir-$<LOWER_CASE:$<CONFIG>>)
    set (MLIR_INSTALL 
      ${CMAKE_BINARY_DIR}/${MLIR_TYPE}-prefix/src/${MLIR_TYPE}/install)
  else()
    set (MLIR_INSTALL ${CMAKE_BINARY_DIR}/mlir_install)

    if (NOT DEFINED LLVM_EXTRA_CMAKE_ARGS)
      set (LLVM_EXTRA_CMAKE_ARGS)
    endif ()
    list (APPEND LLVM_EXTRA_CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${MLIR_INSTALL}
      -DLLVM_ENABLE_PROJECTS=mlir
      -DLLVM_TARGETS_TO_BUILD=X86
      -DLLVM_ENABLE_ASSERTIONS=On
      -DLLVM_ENABLE_RTTI=ON
      -DLLVM_ENABLE_EH=ON
      -DCMAKE_CXX_STANDARD=17
    )

    ExternalProject_Add(mlir
      SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/llvm-project/llvm
      CMAKE_ARGS ${LLVM_EXTRA_CMAKE_ARGS}
      BUILD_ALWAYS true
      USES_TERMINAL_BUILD true
      USES_TERMINAL_CONFIGURE true
    )
    set (MLIR_TARGETS mlir)
  endif()

  # Point Verona at the MLIR
  list (APPEND VERONA_EXTRA_CMAKE_ARGS
    -DVERONA_LLVM_LOCATION=${MLIR_INSTALL}
  )

  # Use top-level install directory when building
  # as a subproject.
  list (APPEND VERONA_EXTRA_CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/dist)

  # Current the ExternalProject_Add_StepTargets seems to be generating something
  # that is breaking CI for MSVC. So on MSVC with have a single build and test
  # target. On other platforms we keep them separate.
  if (MSVC)
    set (SEPARATE_TEST_TARGET false)
  else ()
    set (SEPARATE_TEST_TARGET true)
  endif()

  # Define project level defaults
  list (APPEND VERONA_EXTRA_CMAKE_ARGS
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1
    -DCMAKE_CXX_STANDARD=17)

  ExternalProject_Add(verona 
    SOURCE_DIR ${CMAKE_SOURCE_DIR}
    DEPENDS ${MLIR_TARGETS}
    TEST_COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --timeout 400 --output-on-failure --interactive-debug-mode 0
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

if (VERONA_CI_BUILD)
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

add_subdirectory(external/CLI11)
add_subdirectory(external/fmt)
add_subdirectory(external/pegmatite)

add_library(cpp-peglib INTERFACE)
target_include_directories(cpp-peglib INTERFACE external/cpp-peglib)

add_subdirectory(src)
add_subdirectory(testsuite)

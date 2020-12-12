cmake_minimum_required(VERSION 3.10.0)
project(verona-lang CXX)

include(cmake/clangformat.cmake)
include(cmake/enable-asserts.cmake)

option(VERONA_LLVM_LOCATION "Location of LLVM/MLIR installation")
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

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/dist CACHE
    PATH "Default to installing inside build dir" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

get_property(IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if (IS_MULTI_CONFIG)
  message(STATUS "Build types for Verona: ${CMAKE_CONFIGURATION_TYPES}")
else()
  set(DEFAULT_BUILD_TYPE "Release")
  if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
  endif()
  message(STATUS "Build Type for Verona: ${CMAKE_BUILD_TYPE}")
endif ()

unset(MLIR_DIR CACHE)
unset(LLVM_DIR CACHE)
if (VERONA_LLVM_LOCATION)
  find_package(MLIR REQUIRED CONFIG
    PATHS ${VERONA_LLVM_LOCATION}
    NO_DEFAULT_PATH)
else ()
  include(ExternalProject)

  find_package(Git REQUIRED)

  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short=11 HEAD
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/external/llvm-project"
    OUTPUT_VARIABLE LLVM_GIT_REVISION)
  string(STRIP ${LLVM_GIT_REVISION} LLVM_GIT_REVISION)
  message (STATUS "Detected LLVM GIT commit: ${LLVM_GIT_REVISION}")

  # if (${CMAKE_BUILD_TYPE} STREQUAL RelWithDebInfo)
  #   # RelWithDebInfo should use release LLVM.
  #   set(LLVM_BUILD_TYPE Release)
  # elseif (WIN32)
  #   set(LLVM_BUILD_TYPE ${CMAKE_BUILD_TYPE})
  # else ()
  #   # Use release everywhere it works.
  # endif ()

  set(LLVM_BUILD_TYPE Combined)
  set(LLVM_BASE ${CMAKE_BINARY_DIR}/llvm/${LLVM_BUILD_TYPE}/${LLVM_GIT_REVISION})

  # If LLVM isn't present, and the user runs download-llvm/build-llvm targets,
  # we need to re-run cmake to load MLIR and enable the right targets.
  #
  # We use an llvm-stamp file as configure dependency to automatically run cmake
  # when that happens. Both the download-llvm and build-llvm targets touch that
  # stamp file.
  file(MAKE_DIRECTORY ${LLVM_BASE})
  file(TOUCH ${LLVM_BASE}/llvm-stamp)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${LLVM_BASE}/llvm-stamp)

  if (EXISTS "${LLVM_BASE}/install/lib/cmake/mlir/MLIRConfig.cmake")
    message(STATUS "Using LLVM/MLIR in ${LLVM_BASE}/install/")
    find_package(MLIR REQUIRED CONFIG
      PATHS ${LLVM_BASE}/install
      NO_DEFAULT_PATH)
  else ()
    message(WARNING "Could not find an LLVM installation. The MLIR backend will not be compiled.\nEither define VERONA_LLVM_LOCATION, use the `download-llvm` target to download a prebuilt LLVM blob or use the `build-llvm` to built it from source.")
    set (MLIR_FOUND FALSE)
  endif ()

  add_custom_target(download-llvm
    USES_TERMINAL # This allows us to print progress
    COMMAND
      ${CMAKE_COMMAND}
      -DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}
      -DLLVM_BUILD_TYPE=Debug
      -DLLVM_GIT_REVISION=${LLVM_GIT_REVISION}
      -DVERBOSE_LLVM_DOWNLOAD=${VERBOSE_LLVM_DOWNLOAD}
      -DOUTPUT_DIR=${CMAKE_BINARY_DIR}/llvm/Debug/${LLVM_GIT_REVISION}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/download-llvm.cmake
    
    COMMAND
      ${CMAKE_COMMAND}
      -DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}
      -DLLVM_BUILD_TYPE=Release
      -DLLVM_GIT_REVISION=${LLVM_GIT_REVISION}
      -DVERBOSE_LLVM_DOWNLOAD=${VERBOSE_LLVM_DOWNLOAD}
      -DOUTPUT_DIR=${CMAKE_BINARY_DIR}/llvm/Release/${LLVM_GIT_REVISION}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/download-llvm.cmake

    COMMAND ${CMAKE_COMMAND}
      -DSOURCES="${CMAKE_BINARY_DIR}/llvm/Release/${LLVM_GIT_REVISION}/install\;${CMAKE_BINARY_DIR}/llvm/Debug/${LLVM_GIT_REVISION}/install"
      -DDESTINATION=${CMAKE_BINARY_DIR}/llvm/Combined/${LLVM_GIT_REVISION}/install
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/copy-llvm-libs.cmake

    COMMAND ${CMAKE_COMMAND} -E touch ${LLVM_BASE}/llvm-stamp)

  ExternalProject_Add(build-llvm
    PREFIX ${LLVM_BASE}
    BINARY_DIR ${LLVM_BASE}/build
    STAMP_DIR ${LLVM_BASE}/stamp
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/llvm-project/llvm
    CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${LLVM_BASE}/install
      -DCMAKE_BUILD_TYPE=${LLVM_BUILD_TYPE}
      -DLLVM_ENABLE_PROJECTS=mlir
      -DLLVM_TARGETS_TO_BUILD=X86
      -DLLVM_ENABLE_ASSERTIONS=On
      -DLLVM_ENABLE_RTTI=ON
      -DLLVM_ENABLE_EH=ON
      ${LLVM_EXTRA_CMAKE_ARGS}
    USES_TERMINAL_BUILD true
    USES_TERMINAL_CONFIGURE true
    EXCLUDE_FROM_ALL true
    BUILD_ALWAYS true
  )

  add_custom_command(
    TARGET build-llvm
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E touch ${LLVM_BASE}/llvm-stamp)

endif ()

if (MLIR_FOUND)
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
endif ()

if (VERONA_CI_BUILD)
  # Specify policy to go into child projects.
  set (CMAKE_POLICY_DEFAULT_CMP0077 NEW)
  set (SNMALLOC_CI_BUILD ON)
endif ()

clangformat_targets()

if (ENABLE_ASSERTS)
  enable_asserts()
endif()

if (MSVC)
  add_compile_options(/permissive-)
  add_compile_options(/utf-8)
  add_compile_options(/wd4307)
endif()

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(external/CLI11 EXCLUDE_FROM_ALL)
add_subdirectory(external/fmt EXCLUDE_FROM_ALL)
add_subdirectory(external/pegmatite EXCLUDE_FROM_ALL)

add_library(cpp-peglib INTERFACE)
target_include_directories(cpp-peglib INTERFACE external/cpp-peglib)

enable_testing()

add_subdirectory(src)
add_subdirectory(testsuite)

# Adds a target check that runs the tests.
add_custom_target(check
  COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure --timeout 400 --interactive-debug-mode 0
  USES_TERMINAL
  DEPENDS install)

if (RT_TESTS)
  add_dependencies(check rt_tests)
endif ()

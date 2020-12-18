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

include(cmake/clangformat.cmake)

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

SET(VERONA_LOCAL_DIST ${CMAKE_BINARY_DIR}/dist CACHE PATH "Default to installing inside build dir" FORCE)

# This is to trick the CMake into building LLVM before Verona.
# We use two External Projects, so that the LLVM build can complete and install
# before we start the Verona one.  This is required as we consume CMake
# artifacts, which are not available otherwise.
include(ExternalProject)

project(verona CXX)
set(DEFAULT_BUILD_TYPE "Release")
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
endif()

if (VERONA_DOWNLOAD_LLVM AND DEFINED LLVM_EXTRA_CMAKE_ARGS)
  message (WARNING "Ignoring LLVM_EXTRA_CMAKE_ARGS as using downloaded LLVM.")
endif ()

if (VERONA_DOWNLOAD_LLVM AND DEFINED VERONA_EXTERNAL_LLVM_DIR)
  message (WARNING "Ignoring VERONA_EXTERNAL_LLVM_DIR as using downloaded LLVM.")
endif ()

##########################################################
#  Configure MLIR installation
##########################################################
if (DEFINED VERONA_EXTERNAL_LLVM_DIR)
  # Use existing installation from system
  set (MLIR_INSTALL ${VERONA_EXTERNAL_LLVM_DIR})
  # Fake target for MLIR, it is built externally.
  add_custom_target(external)
else()
  #  Build or Download MLIR installation
  set (MLIR_INSTALL ${CMAKE_BINARY_DIR}/$<CONFIG>/mlir)
  set (EXTERNAL_EXTRA_CMAKE_ARGS)
  list (APPEND EXTERNAL_EXTRA_CMAKE_ARGS 
    -DVERONA_DOWNLOAD_LLVM=${VERONA_DOWNLOAD_LLVM}
    -DMLIR_INSTALL=${MLIR_INSTALL}
    -DCMAKE_BUILD_TYPE=$<CONFIG>)

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
endif()

##########################################################
#  Build Verona subproject
##########################################################
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

# Use top-level install directory required for tests.
list (APPEND VERONA_EXTRA_CMAKE_ARGS
  -DCMAKE_INSTALL_PREFIX=${VERONA_LOCAL_DIST})
list (APPEND VERONA_DONT_PASS CMAKE_INSTALL_PREFIX)

# Set up variables to pass arguments to sub project
get_cmake_property(CACHE_VARS CACHE_VARIABLES)
foreach (CACHE_VAR ${CACHE_VARS})
  list (FIND VERONA_DONT_PASS ${CACHE_VAR} INDEX)
  if (${INDEX} EQUAL -1)
    list(APPEND VERONA_EXTRA_CMAKE_ARGS
      -D${CACHE_VAR}=${${CACHE_VAR}}
    )
  endif()
endforeach ()

ExternalProject_Add(verona
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/src
  DEPENDS external
  BUILD_ALWAYS true
  USES_TERMINAL_BUILD true
  USES_TERMINAL_CONFIGURE true
  USES_TERMINAL_INSTALL true
  USES_TERMINAL_TEST true
  CMAKE_ARGS ${VERONA_EXTRA_CMAKE_ARGS})

######################################################
#  Add testing at top level
######################################################
enable_testing()

include(ProcessorCount)
ProcessorCount(N)

add_subdirectory(testsuite)

# Adds a target check that runs the tests.
add_custom_target(check DEPENDS verona)
add_custom_command(TARGET check 
  COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> -j ${N} --output-on-failure --timeout 400 --interactive-debug-mode 0
  USES_TERMINAL
)

# Main target does not test runtime, build subdirectory for that
# and add to check target.
if (RT_TESTS)
  add_subdirectory(src/rt)
endif()

return ()

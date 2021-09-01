cmake_minimum_required(VERSION 3.10.0)

project(verona CXX)

# Option that doesn't get passed to the VeronaC ExternalProject
set (VERONA_DONT_PASS)
macro (set_cache_top VAR HELP TYPE DEFAULT)
  set (${VAR} ${DEFAULT} CACHE ${TYPE} ${HELP})
  list (APPEND VERONA_DONT_PASS ${VAR})
endmacro()
macro (option_top VAR HELP DEFAULT)
  set_cache_top(${VAR} ${HELP} BOOL ${DEFAULT})
endmacro ()

# Default is to download LLVM blob, this shows progress
option_top(VERBOSE_LLVM_DOWNLOAD "Verbose LLVM/LLVM download step" OFF)

# Use external LLVM directory options
set_cache_top(VERONA_USE_EXTERNAL_LLVM_DIR "Location of an external LLVM installation to use. Do not set with DOWNLOAD LLVM / LLVM SUBMODULE options." PATH "")

# Build LLVM from submodule
option_top(VERONA_BUILD_LLVM_SUBMODULE "Force building the LLVM submodule. Do not set with DOWNLOAD LLVM / EXTERNAL LLVM options." OFF)
set_cache_top(LLVM_EXTRA_CMAKE_ARGS "Additional options for building LLVM submodule. Do not set with DOWNLOAD LLVM / EXTERNAL LLVM options." STRING "")

# Other Verona options
option(ENABLE_ASSERTS "Enable asserts even in release builds" OFF)
option_top(RT_TESTS "Including unit tests for the runtime" OFF)
option_top(VERONA_EXPENSIVE_SYSTEMATIC_TESTING "Increase the range of seeds covered by systematic testing" OFF)
option(USE_SCHED_STATS "Track scheduler stats" OFF)
option(VERONA_CI_BUILD "Disable features not sensible for CI" OFF)
option(USE_SYSTEMATIC_TESTING "Enable systematic testing in the runtime" OFF)
option(USE_CRASH_LOGGING "Enable crash logging in the runtime" OFF)
if (NOT MSVC)
  option(CMAKE_EXPORT_COMPILE_COMMANDS "Export compilation commands" ON)
endif ()

set(SANITIZER "" CACHE STRING "Use sanitizer type (address|undefined|memory|...)")
if (SANITIZER)
  message(STATUS "Using sanitizer=${SANITIZER}")
endif()

##########################################################
# Pass parameters to subbuild.
##########################################################
# CMAKE_HOME_DIRECTORY should not be passed down to another project.
list (APPEND VERONA_DONT_PASS CMAKE_HOME_DIRECTORY)
# We override the Verona Install prefix to build a local distribution for tests.
list (APPEND VERONA_DONT_PASS CMAKE_INSTALL_PREFIX)

if (DEFINED VERONA_EXTRA_CMAKE_ARGS)
  message (WARNING "Do not use VERONA_EXTRA_CMAKE_ARGS! Value ignored")
endif ()
set (VERONA_EXTRA_CMAKE_ARGS)
# Pass parameters to the subproject

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

## Clang-format target
include(cmake/clangformat.cmake)
clangformat_targets()

## Check copyright/license target
include(cmake/copyright.cmake)
copyright_targets()

# ExternalProject is used to trick the CMake into building LLVM before Verona.
# We use two External Projects, so that the LLVM build can complete and install
# before we start the Verona one.  This is required as we consume CMake
# artifacts, which are not available otherwise.
include(ExternalProject)

set(DEFAULT_BUILD_TYPE "Release")
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
endif()

# LLVM Options
#
# Each option below works on their own and are incompatible with each other.
#  * VERONA_DOWNLOAD_LLVM (Default=ON): Downloads a pre-build LLVM for Verona.
#    (+ VERBOSE_LLVM_DOWNLOAD: Shows download progress (Default=OFF))
#  * VERONA_USE_EXTERNAL_LLVM_DIR: The path of a pre-built install dir of an LLVM toolchain.
#  * VERONA_BUILD_LLVM_SUBMODULE: Builds from the submodule.
#    (+ LLVM_EXTRA_CMAKE_ARGS: Extra CMake arguments to build LLVM)
#
# Note: On both download and build options, the LLVM checkout are guaranteed to
# be the same as the submodule, and have been validated by CI. For the external
# build, users need to make sure they either get the same commit or they know what
# they're doing.
if (VERONA_USE_EXTERNAL_LLVM_DIR OR (VERONA_BUILD_LLVM_SUBMODULE OR LLVM_EXTRA_CMAKE_ARGS))
  if (VERONA_USE_EXTERNAL_LLVM_DIR AND (VERONA_BUILD_LLVM_SUBMODULE OR LLVM_EXTRA_CMAKE_ARGS))
    message (SEND_ERROR "Incompatible options VERONA_BUILD_LLVM_SUBMODULE/LLVM_EXTRA_CMAKE_ARGS and VERONA_USE_EXTERNAL_LLVM_DIR.")
  elseif (LLVM_EXTRA_CMAKE_ARGS AND NOT VERONA_BUILD_LLVM_SUBMODULE)
    message (SEND_ERROR "Invalid use of LLVM_EXTRA_CMAKE_ARGS without VERONA_BUILD_LLVM_SUBMODULE.")
  else()
    if (VERONA_USE_EXTERNAL_LLVM_DIR)
      message (STATUS "Import LLVM from: ${VERONA_USE_EXTERNAL_LLVM_DIR}")
    elseif(VERONA_BUILD_LLVM_SUBMODULE)
      if (LLVM_EXTRA_CMAKE_ARGS)
        message (STATUS "Build LLVM with args: ${LLVM_EXTRA_CMAKE_ARGS}")
      else()
        message (STATUS "Build LLVM: ${VERONA_BUILD_LLVM_SUBMODULE}")
      endif()
    endif()
  endif ()
  # Download is default, so if we pick one of these, we want to turn off download
  set(VERONA_DOWNLOAD_LLVM OFF)
else()
  set(VERONA_DOWNLOAD_LLVM ON)
  message (STATUS "Download LLVM: ${VERONA_DOWNLOAD_LLVM}")
endif()

##########################################################
#  Configure LLVM installation
##########################################################

if(VERONA_USE_EXTERNAL_LLVM_DIR)
  # Fake target for LLVM, it is built externally.
  add_custom_target(external)
  # Use existing installation from system
  list (APPEND VERONA_EXTRA_CMAKE_ARGS
    -DVERONA_LLVM_LOCATION=${VERONA_USE_EXTERNAL_LLVM_DIR}
  )
else()
  set(LLVM_INSTALL ${CMAKE_BINARY_DIR}/$<CONFIG>/mlir)
  #  Build or Download LLVM installation
  set (EXTERNAL_EXTRA_CMAKE_ARGS)
  list (APPEND EXTERNAL_EXTRA_CMAKE_ARGS 
    -DVERONA_DOWNLOAD_LLVM=${VERONA_DOWNLOAD_LLVM}
    -DLLVM_INSTALL=${LLVM_INSTALL}
    -DCMAKE_BUILD_TYPE=$<CONFIG>)

  # Extra options for building or downloading LLVM
  if(VERONA_BUILD_LLVM_SUBMODULE)
    list (APPEND EXTERNAL_EXTRA_CMAKE_ARGS
      -DLLVM_EXTRA_CMAKE_ARGS=${LLVM_EXTRA_CMAKE_ARGS}
    )
  elseif(VERONA_DOWNLOAD_LLVM)
    list (APPEND EXTERNAL_EXTRA_CMAKE_ARGS
      -DVERBOSE_LLVM_DOWNLOAD=${VERBOSE_LLVM_DOWNLOAD}
    )
  endif()

  ExternalProject_Add(external
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/external
    CMAKE_ARGS ${EXTERNAL_EXTRA_CMAKE_ARGS} 
    BUILD_ALWAYS true
    INSTALL_COMMAND ""
    TEST_COMMAND ""
    USES_TERMINAL_BUILD true
    USES_TERMINAL_CONFIGURE true
  )

  # Point Verona at the LLVM install, we just built/downloaded
  list (APPEND VERONA_EXTRA_CMAKE_ARGS
    -DVERONA_LLVM_LOCATION=${CMAKE_BINARY_DIR}/$<CONFIG>/mlir/install
  )
endif()


##########################################################
#  Build Verona subproject
##########################################################
SET(VERONA_LOCAL_DIST ${CMAKE_BINARY_DIR}/dist)

# Use top-level install directory required for tests.
list (APPEND VERONA_EXTRA_CMAKE_ARGS
  -DCMAKE_INSTALL_PREFIX=${VERONA_LOCAL_DIST}/)

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

include(cmake/enable-asserts.cmake)
if (ENABLE_ASSERTS)
  enable_asserts()
endif()

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
  add_dependencies(check rt_tests)
endif()

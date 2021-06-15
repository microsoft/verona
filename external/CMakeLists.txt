# Build the external dependencies that need to be built as external projects
#
# This layer of cmake is guaranteed to have a CMAKE_BUILD_TYPE set, as the
# calling context will have passed $<CONFIG> as an argument. 
# This enables better interaction with downloads and multi-config builds such as
# msbuild.  It also means that changing build type repeatedly can used cached
# versions of external artifacts.
cmake_minimum_required(VERSION 3.10.0)

project(verona_external CXX)
include(ExternalProject)

message (STATUS "Install MLIR to ${MLIR_INSTALL}")

if (VERONA_DOWNLOAD_LLVM)
  find_package(Git)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short=11 HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/llvm-project"
    OUTPUT_VARIABLE LLVM_PACKAGE_GIT_VERSION
  )
  string(STRIP ${LLVM_PACKAGE_GIT_VERSION} LLVM_PACKAGE_GIT_VERSION)
  message (STATUS "Detected GIT commit: ${LLVM_PACKAGE_GIT_VERSION}")

  string(TOLOWER ${CMAKE_SYSTEM_NAME} PLATFORM)
  # Need loop to download all types for possible configurations
  # Required by multi-build systems like msbuild
  string (TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
  if (${BUILD_TYPE} STREQUAL relwithdebinfo)
    # RelWithDebInfo should use release LLVM.
    set(LLVM_BUILD_TYPE release)
  elseif (WIN32)
    set(LLVM_BUILD_TYPE ${BUILD_TYPE})
  else ()
    # Use release everywhere it works.
    set(LLVM_BUILD_TYPE release)
  endif ()


  # Setup LLVM download (+MD5 string)
  set(LLVM_URL https://verona.blob.core.windows.net/llvmbuild)
  set(PKG_NAME verona-llvm-install-x86_64-${PLATFORM}-${LLVM_BUILD_TYPE}-${LLVM_PACKAGE_GIT_VERSION})
  set(MD5_NAME ${PKG_NAME}.md5)
  file(DOWNLOAD "${LLVM_URL}/${MD5_NAME}" ${CMAKE_BINARY_DIR}/${MD5_NAME} STATUS MD5_DOWNLOAD_STATUS)
  list(GET MD5_DOWNLOAD_STATUS 0 MD5_DOWNLOAD_STATUS_CODE)
  if (NOT (${MD5_DOWNLOAD_STATUS_CODE} EQUAL 0))
    list(GET MD5_DOWNLOAD_STATUS 1 ERROR_MESSAGE)
    message(FATAL_ERROR "Failed to download md5 hash: ${ERROR_MESSAGE}")
  endif ()
  file(STRINGS ${CMAKE_BINARY_DIR}/${MD5_NAME} LLVM_MD5_SUM REGEX [0-9a-f]+)
  string(STRIP ${LLVM_MD5_SUM} LLVM_MD5_SUM)

  ExternalProject_Add(mlir
    URL ${LLVM_URL}/${PKG_NAME}
    URL_MD5 ${LLVM_MD5_SUM}
    DOWNLOAD_NO_PROGRESS NOT ${VERBOSE_LLVM_DOWNLOAD}
    CONFIGURE_COMMAND ""
    PREFIX mlir-${LLVM_BUILD_TYPE}
    SOURCE_DIR ${MLIR_INSTALL}
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    TEST_COMMAND ""
    USES_TERMINAL_DOWNLOAD true
  )
else()
  separate_arguments(LLVM_EXTRA_CMAKE_ARGS UNIX_COMMAND ${LLVM_EXTRA_CMAKE_ARGS})
  list (APPEND LLVM_EXTRA_CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${MLIR_INSTALL}/install  # Add install as tar has install prefix. Outer project knows about this.
    # Note: We must use the SEMICOLON generator expression here because CMake
    # uses semicolons as list separators and this must be a literal semicolon
    # in the final output.  There are at least two steps in between this line
    # and the final command-line invocation that parse the string as a list and
    # so any amount of escaping is fragile.
    -DLLVM_ENABLE_PROJECTS=clang$<SEMICOLON>mlir
    -DLLVM_TARGETS_TO_BUILD=X86
    -DLLVM_ENABLE_ASSERTIONS=On
    -DLLVM_ENABLE_RTTI=ON
    -DLLVM_ENABLE_EH=ON
    -DCMAKE_CXX_STANDARD=17
  )

  ExternalProject_Add(mlir
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/llvm-project/llvm
    CMAKE_ARGS ${LLVM_EXTRA_CMAKE_ARGS}
    BUILD_ALWAYS true
    USES_TERMINAL_BUILD true
    USES_TERMINAL_CONFIGURE true
  )
endif()

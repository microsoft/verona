# Download an appropriate prebuilt LLVM binary.
# This script is exposed as the `download-llvm` target by the top-level
# CMakeLists.txt, using cmake's -P script functionality.
#
# The script expects the following variables to be defined:
# - CMAKE_SYSTEM_NAME: The target operating system, ie. linux, windows or
#                      darwin. This is copied verbatim from CMake's definition.
#
# - LLVM_BUILD_TYPE:   The build type for LLVM. These correspond to the build
#                      types produced by the `ci/llvm.yml` file. Currently, only
#                      Debug and Release blobs are produced.
#
# - LLVM_GIT_REVISION: The LLVM git revision to use. We use the revision of the
#                      external/llvm-project submodule to pick this.
#
# - OUTPUT_DIR:        Where to place the result. LLVM will be installed to
#                      ${OUTPUT_DIR}/install. The directory is also used to
#                      store the intermediate files, such as the tarball.

if(NOT DEFINED CMAKE_SYSTEM_NAME
    OR NOT DEFINED LLVM_BUILD_TYPE
    OR NOT DEFINED LLVM_GIT_REVISION
    OR NOT DEFINED OUTPUT_DIR)
  message(FATAL_ERROR "Variables CMAKE_SYSTEM_NAME, LLVM_BUILD_TYPE, LLVM_GIT_REVISION and OUTPUT_DIR must be provided.")
endif()

string(TOLOWER ${LLVM_BUILD_TYPE} BUILD_TYPE)
string(TOLOWER ${CMAKE_SYSTEM_NAME} PLATFORM)

set(LLVM_URL https://verona.blob.core.windows.net/llvmbuild)
set(PKG_NAME verona-llvm-install-x86_64-${PLATFORM}-${BUILD_TYPE}-${LLVM_GIT_REVISION})
set(MD5_NAME ${PKG_NAME}.md5)
set(ARCHIVE_NAME ${PKG_NAME}.tar.gz)

message(STATUS "Downloading LLVM checksum at ${LLVM_URL}/${MD5_NAME}")

file(DOWNLOAD "${LLVM_URL}/${MD5_NAME}" ${OUTPUT_DIR}/${MD5_NAME} STATUS MD5_DOWNLOAD_STATUS)
list(GET MD5_DOWNLOAD_STATUS 0 MD5_DOWNLOAD_STATUS_CODE)
if (NOT (${MD5_DOWNLOAD_STATUS_CODE} EQUAL 0))
  list(GET MD5_DOWNLOAD_STATUS 1 ERROR_MESSAGE)
  message(FATAL_ERROR "Failed to download md5 hash: ${ERROR_MESSAGE}")
endif ()

file(STRINGS ${OUTPUT_DIR}/${MD5_NAME} LLVM_MD5_SUM REGEX [0-9a-f]+)
string(STRIP ${LLVM_MD5_SUM} LLVM_MD5_SUM)

message(STATUS "Downloading LLVM at ${LLVM_URL}/${PKG_NAME}")

if (VERBOSE_LLVM_DOWNLOAD)
  file(DOWNLOAD
    "${LLVM_URL}/${PKG_NAME}"
    "${OUTPUT_DIR}/${ARCHIVE_NAME}"
    SHOW_PROGRESS
    EXPECTED_HASH MD5=${LLVM_MD5_SUM})
else()
  file(DOWNLOAD
    "${LLVM_URL}/${PKG_NAME}"
    "${OUTPUT_DIR}/${ARCHIVE_NAME}"
    EXPECTED_HASH MD5=${LLVM_MD5_SUM})
endif()

message(STATUS "Extracting LLVM")

file(ARCHIVE_EXTRACT
  INPUT "${OUTPUT_DIR}/${ARCHIVE_NAME}"
  DESTINATION ${OUTPUT_DIR}/${PKG_NAME})

# All files in the archive are in a build/install subdirectory.
# We need to move that directory to ${OUTPUT_DIR}/install, which is where the
# main CMakeLists.txt expects it to be.
if (EXISTS ${OUTPUT_DIR}/install)
  file(REMOVE_RECURSE ${OUTPUT_DIR}/install)
endif ()
file(RENAME ${OUTPUT_DIR}/${PKG_NAME}/build/install ${OUTPUT_DIR}/install)
file(REMOVE_RECURSE ${OUTPUT_DIR}/${PKG_NAME})

# TODO: This is only temporary; it can be removed by rebuilding the LLVM blobs
# with -DCMAKE_DEBUG_POSTFIX="-debug".
if (BUILD_TYPE STREQUAL "debug")
  message(STATUS "Patching debug configuration.")

  file(GLOB LIBS ${OUTPUT_DIR}/install/lib/*.a)
  foreach(original_name ${LIBS})
    string(REGEX REPLACE "\\.(a|lib|dylib|dll|so)$" "-debug.\\1" patched_name ${original_name})
    file(RENAME ${original_name} ${patched_name})
  endforeach()

  execute_process(
    COMMAND sed -i "s#\\(\\${_IMPORT_PREFIX}/lib/.*\\)\\.\\(a\\|lib|dll|dylib|so\\)\\b#\\1-debug.\\2#g"
      ${OUTPUT_DIR}/install/lib/cmake/mlir/MLIRTargets-debug.cmake
      ${OUTPUT_DIR}/install/lib/cmake/llvm/LLVMExports-debug.cmake)
endif ()

message(STATUS "Done.")

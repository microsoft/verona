# Define FIXED_VCPKG_VERSION to provide a fixed commit of vcpkg to use.  This
# can be used by CI to ensure that we're able to use the versions from the
# cache.
if (NOT DEFINED FIXED_VCPKG_VERSION)
    set(FIXED_VCPKG_VERSION master)
	set(FIXED_VCPKG_ZIP refs/heads/master)
else()
	set(FIXED_VCPKG_ZIP ${FIXED_VCPKG_VERSION})
endif ()

# Not used yet, but we may want to add custom triplets for ASAN and so on.
#set("VCPKG_OVERLAY_TRIPLETS" "${CMAKE_SOURCE_DIR}/overlay-triplets")
set("VCPKG_OVERLAY_PORTS" "${CMAKE_SOURCE_DIR}/overlay-ports")
set("VERONA_DEFAULT_VCPKG_LOCATION" "${CMAKE_CURRENT_BINARY_DIR}/vcpkg-${FIXED_VCPKG_VERSION}/scripts/buildsystems/vcpkg.cmake")
if (NOT DEFINED VCPKG_BOOTSTRAP_OPTIONS)
  message(STATUS "vcpkg will default to using bundled tools when bootstrapping.  If this doens't work on your platform please re-run cmake with -DVCPKG_BOOTSTRAP_OPTIONS=-useSystemBinaries")
endif()
# Fetch vcpkg from GitHub
if (NOT EXISTS ${VERONA_DEFAULT_VCPKG_LOCATION})
  set(DOWNLOAD_PATH "https://github.com/microsoft/vcpkg/archive/${FIXED_VCPKG_ZIP}.zip")
  message(STATUS "Vcpkg not found, downloading from ${DOWNLOAD_PATH}")
  file(DOWNLOAD ${DOWNLOAD_PATH} vcpkg.zip
    SHOW_PROGRESS
    TLS_VERIFY ON)
  file(ARCHIVE_EXTRACT INPUT vcpkg.zip)
endif()
# Use vcpkg as the toolchain file.  This will bootstrap vcpkg.
set(CMAKE_TOOLCHAIN_FILE ${VERONA_DEFAULT_VCPKG_LOCATION}
    CACHE STRING "Vcpkg toolchain file")

if (NOT EXISTS ${VERONA_DEFAULT_VCPKG_LOCATION})
	message(FATAL_ERROR "vcpkg toolchain file not found: ${VERONA_DEFAULT_VCPKG_LOCATION}")
endif()

cmake_minimum_required(VERSION 3.14.0)
project(verona VERSION 1.0.0 LANGUAGES CXX)

include(FetchContent)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(TRIESTE_BUILD_SAMPLES OFF)

option(VERONA_GENERATE_LAUNCH_JSON "Generate a launch.json file for debugging in VSCode" OFF)

FetchContent_Declare(
  trieste
  GIT_REPOSITORY https://github.com/microsoft/trieste
  GIT_TAG b466068270471ccc9c5f5ddd543bd6e2fb02ad87
  GIT_SHALLOW FALSE
)

FetchContent_MakeAvailable(trieste)

# Use snmalloc clangformat target
clangformat_targets()

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG 9.1.0
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(fmt)

set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

# Installation instructions
set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/dist)

# Clear all existing files and folders from the install directory
install(CODE [[
  file(REMOVE_RECURSE ${CMAKE_INSTALL_PREFIX}/.)
  ]])

# Copy the stdlib to the install directory
install(DIRECTORY std DESTINATION verona)

enable_testing()
add_subdirectory(src)
add_subdirectory(testsuite)
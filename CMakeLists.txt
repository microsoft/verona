cmake_minimum_required(VERSION 3.14.0)
project(verona VERSION 1.0.0 LANGUAGES CXX)

include(FetchContent)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(TRIESTE_BUILD_SAMPLES OFF)

FetchContent_Declare(
  trieste
  GIT_REPOSITORY https://github.com/microsoft/trieste
  GIT_TAG main
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(trieste)

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

enable_testing()
add_subdirectory(src)

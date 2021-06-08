cmake_minimum_required(VERSION 3.10.0)

include_directories(.)

project(verona-lang CXX)

set(CMAKE_CXX_STANDARD 17)

message(STATUS "Build Type for Verona ${CMAKE_BUILD_TYPE}")
set(CMAKE_CONFIGURATION_TYPES  Release Debug RelWithDebInfo)
message(STATUS "Build types ${CMAKE_CONFIGURATION_TYPES}")

if (VERONA_CI_BUILD)
  # Specify policy to go into child projects.
  set (CMAKE_POLICY_DEFAULT_CMP0077 NEW)
  set (SNMALLOC_CI_BUILD ON)
endif ()

include(../cmake/enable-asserts.cmake)
if (ENABLE_ASSERTS)
  enable_asserts()
endif()

add_subdirectory(../external/CLI11 ./external/CLI11 EXCLUDE_FROM_ALL)
add_subdirectory(../external/fmt ./external/fmt EXCLUDE_FROM_ALL)
add_subdirectory(../external/pegmatite ./external/pegmatite EXCLUDE_FROM_ALL)

add_library(cpp-peglib INTERFACE)
target_include_directories(cpp-peglib INTERFACE ../external/cpp-peglib)

if(MSVC)
  add_compile_options(/permissive-)
  add_compile_options(/utf-8)
  add_compile_options(/wd4307)
  add_compile_options(/wd4267)
  add_compile_options(/wd4291)
  add_compile_options(/wd4624)
endif()

set (VERONA_RT_ONLY_HEADER_LIBRARY ON)
add_subdirectory(rt)

# Should go to stricter warnings 
#   warnings_high()
# But for now, just add some to stop the Windows build breaking
if(MSVC)
  add_compile_options(/WX)
endif()
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  add_compile_options(-Wmismatched-tags)
endif()

add_subdirectory(parser)
add_subdirectory(compiler)
add_subdirectory(interpreter)
add_subdirectory(mlir)
add_subdirectory(interop)

install(CODE [[
  file(REMOVE_RECURSE ${CMAKE_INSTALL_PREFIX})
  ]])
install(DIRECTORY stdlib/ DESTINATION stdlib
  PATTERN "CMakeLists.txt" EXCLUDE)

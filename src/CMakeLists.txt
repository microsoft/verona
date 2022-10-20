cmake_minimum_required(VERSION 3.18.0)

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

# Sanitizers must be added first, or it gets upset.
if(SANITIZER)
  if(MSVC)
    message(FATAL_ERROR "MSVC does not support sanitizers")
  endif()
  message(STATUS "Using sanitizer=${SANITIZER}")
  add_compile_options(-fsanitize=${SANITIZER})
  add_link_options(-fsanitize=${SANITIZER})
  # Disable vptr check in UBSAN (incompatible with -no-rtti used by LLVM)
  # https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html#available-checks
  if (SANITIZER MATCHES "undefined")
    add_compile_options(-fno-sanitize=vptr)
    add_link_options(-fno-sanitize=vptr)
  endif()
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


include(FetchContent)

FetchContent_Declare(
  verona_rt
  GIT_REPOSITORY https://github.com/microsoft/verona-rt
  GIT_TAG        aa1cab8c9c7cbc1e55e11d0464c0ea38ff9fe6d1
  SOURCE_SUBDIR  src
)

FetchContent_MakeAvailable(verona_rt)

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
add_subdirectory(deprecated)
add_subdirectory(mlir)
add_subdirectory(interop)

install(CODE [[
  file(REMOVE_RECURSE ${CMAKE_INSTALL_PREFIX})
  ]])
install(DIRECTORY stdlib/ DESTINATION stdlib
  PATTERN "CMakeLists.txt" EXCLUDE)

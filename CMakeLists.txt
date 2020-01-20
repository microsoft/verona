cmake_minimum_required(VERSION 3.10.0)
project(verona-lang CXX)

set(CMAKE_CXX_STANDARD 17)

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/dist CACHE PATH "Default to installing inside build dir" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

enable_testing()

if(MSVC)
  add_compile_options(/permissive-)
  add_compile_options(/utf-8)
endif()

add_subdirectory(external/CLI11)
add_subdirectory(external/fmt)
add_subdirectory(external/pegmatite)

add_subdirectory(src)
add_subdirectory(testsuite)

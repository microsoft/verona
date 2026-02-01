# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.15)

include(FetchContent)

# Used to provide
#  FetchContent_MakeAvailable_ExcludeFromAll
FetchContent_Declare(
    cmake_utils
    GIT_REPOSITORY https://github.com/mjp41/cmake_utils
    GIT_TAG 2bf98b5773ea7282197c823e205547d8c2e323c0
    GIT_SHALLOW FALSE
)

FetchContent_MakeAvailable(cmake_utils)


FetchContent_Declare(
  trieste
  GIT_REPOSITORY https://github.com/mjp41/trieste
  GIT_TAG        infix-wip
)

set(TRIESTE_BUILD_SAMPLES OFF)
set(TRIESTE_BUILD_PARSERS OFF)
set(TRIESTE_ENABLE_TESTING OFF)
FetchContent_MakeAvailable_ExcludeFromAll(trieste EXCLUDE_FROM_ALL)

project(infix_lang)

set (SOURCES 
    src/main.cc 
    src/parser.cc 
    src/passes/pass.cc 
    src/passes/lookup_util.cc 
    src/passes/operator_defn.cc 
    src/passes/function_parse.cc 
    src/passes/parse_types.cc 
    src/passes/resolve_types.cc 
    src/passes/infix_parse.cc
)

add_executable(infixlang ${SOURCES})

target_link_libraries(infixlang PRIVATE trieste)

target_include_directories(infixlang PRIVATE src)

clangformat_targets()

enable_testing()
add_subdirectory(testsuite)
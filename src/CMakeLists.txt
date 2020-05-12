include_directories(.)

option(RT_TESTS "Including unit tests for the runtime" OFF)

# Main target does not test runtime, build subdirectory for that
if (RT_TESTS)
else()
  set (VERONA_RT_ONLY_HEADER_LIBRARY ON)
endif()
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


add_subdirectory(ast)
add_subdirectory(compiler)
add_subdirectory(interpreter)
add_subdirectory(stdlib)

# Some builds (ex. clang format) don't need LLVM
if(NOT DISABLE_LLVM_BUILD)
  add_subdirectory(mlir)
endif()

clangformat_targets()

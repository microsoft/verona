# Some builds (ex. clang format) don't need external dependencies like LLVM
if(VERONA_MINIMAL_CMAKE)
  # Lifted from snmalloc. Hard to include with external projects, so copied
  macro(clangformat_targets)
    # The clang-format tool is installed under a variety of different names.  Try
    # to find a sensible one.  Only look for versions 9 explicitly - we don't
    # know whether our clang-format file will work with newer versions of the
    # tool.  It does not work with older versions as AfterCaseLabel is not supported
    # in earlier versions.
    find_program(CLANG_FORMAT NAMES
      clang-format-9)

    # If we've found a clang-format tool, generate a target for it, otherwise emit
    # a warning.
    if (${CLANG_FORMAT} STREQUAL "CLANG_FORMAT-NOTFOUND")
      message(WARNING "Not generating clangformat target, no clang-format tool found")
    else ()
      message(STATUS "Generating clangformat target using ${CLANG_FORMAT}")
      file(GLOB_RECURSE ALL_SOURCE_FILES *.cc *.h *.hh)
      add_custom_target(
        clangformat
        COMMAND ${CLANG_FORMAT}
        -i
        ${ALL_SOURCE_FILES})
    endif()
  endmacro()

  clangformat_targets()
  return()
endif()

include_directories(.)

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
add_subdirectory(mlir)
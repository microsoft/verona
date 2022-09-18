# Lifted from snmalloc. Hard to include with external projects, so copied
macro(clangformat_targets)
  # The clang-format tool is installed under a variety of different names.  Try
  # to find a sensible one.  Only look for versions 9 explicitly - we don't
  # know whether our clang-format file will work with newer versions of the
  # tool.  It does not work with older versions as AfterCaseLabel is not supported
  # in earlier versions.
  #
  # This can always be overridden with `-DCLANG_FORMAT=/path/to/clang-format` if
  # need be.
  find_program(CLANG_FORMAT NAMES
    clang-format-9
    clang-format90)

  # If we've found a clang-format tool, generate a target for it, otherwise emit
  # a warning.
  if (${CLANG_FORMAT} STREQUAL "CLANG_FORMAT-NOTFOUND")
    message(WARNING "Not generating clangformat target, must have clang-format-9 in the PATH")
  else ()
    message(STATUS "Generating clangformat target using ${CLANG_FORMAT}")
    find_package(Git)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} ls-files *.cc *.h
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
      OUTPUT_VARIABLE ALL_SOURCE_FILES_STRING
    )

    string(REPLACE "\n" ";" ALL_SOURCE_FILES ${ALL_SOURCE_FILES_STRING})

    add_custom_target(
      clangformat
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
      COMMAND ${CLANG_FORMAT}
      -i
      ${ALL_SOURCE_FILES})
  endif()
endmacro()

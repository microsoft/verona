macro(copyright_targets)
  # This macro finds all sources and checks that they all have the correct
  # copyright headers. Needs bash, grep, tee, xargs, git.
  find_program(BASH NAMES bash)

  message(STATUS "Generating copyright/license target")
  add_custom_target(
    copyright
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
    COMMAND bash utils/copyright_check.sh)
endmacro()

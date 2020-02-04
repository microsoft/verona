# Included by each mode script to provide some common functionality.
# The following variables are expected to be defined:
# - PYTHON_EXECUTABLE: Path to the Python interpreter
# - VERONAC: Path to the Verona compiler
# - FILECHECK: Path to LLVM's FileCheck binary
# - CHECK_DUMP_PY: Path to the check_dump.py script in verona-lang/utils
# - TEST_NAME: Full name of the test (e.g. resolution/compile-pass/circular)
# - TEST_FILE: Path to the test source file

if (NOT EXISTS ${VERONAC})
  message(FATAL_ERROR " To run tests you must build the install target." ${VERONAC})
endif ()


# Do some discovery about auxiliary files, setup a few directories
function(PrepareTest _VERONAC_FLAGS _EXPECTED_DUMP _ACTUAL_DUMP)
  get_filename_component(parentdir ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME} DIRECTORY)
  file(MAKE_DIRECTORY ${parentdir})

  set(EXPECTED_DUMP ${SOURCE_DIR}/${TEST_NAME})
  set(ACTUAL_DUMP ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}.dump)

  if(IS_DIRECTORY "${EXPECTED_DUMP}")
    # Create the dump directory (if it doesn't yet exist) and cleanup any file
    # from previous runs.
    file(MAKE_DIRECTORY ${ACTUAL_DUMP})
    file(GLOB old_dump_files ${ACTUAL_DUMP}/*)
    if(old_dump_files)
      file(REMOVE ${old_dump_files})
    endif()

    set(${_EXPECTED_DUMP} ${EXPECTED_DUMP} PARENT_SCOPE)
    set(${_ACTUAL_DUMP} ${ACTUAL_DUMP} PARENT_SCOPE)
    set(${_VERONAC_FLAGS} ${${_VERONAC_FLAGS}} --dump-path=${ACTUAL_DUMP} PARENT_SCOPE)
  endif()
endfunction()

# Execute a command and check its return code against a given value.
#
# Example use:
#   CheckStatus(COMMAND ./binary arg EXPECTED_STATUS 1)
#
function(CheckStatus)
  cmake_parse_arguments(
    PARSE_ARGV 0
    CHECK_STATUS
    ""
    "EXPECTED_STATUS"
    "COMMAND")

  string (REPLACE ";" " " cmd_str "${CHECK_STATUS_COMMAND}")
  message(STATUS "Executing \"${cmd_str}\"")
  execute_process(
    COMMAND ${CHECK_STATUS_COMMAND}
    RESULT_VARIABLE code
    ${CHECK_STATUS_UNPARSED_ARGUMENTS})

  if(NOT ${code} EQUAL ${CHECK_STATUS_EXPECTED_STATUS})
    message(FATAL_ERROR " \"${cmd_str}\" exited with error code ${code}, expected ${CHECK_STATUS_EXPECTED_STATUS}")
  else()
    message(STATUS "Executing \"${cmd_str}\" completed!")
  endif()
endfunction()

# Run the check_dump.py script, comparing two directories
function(CheckDump expected actual)
  CheckStatus(
    COMMAND ${PYTHON_EXECUTABLE} ${CHECK_DUMP_PY} ${expected} ${actual}
    EXPECTED_STATUS 0)
endfunction()

# Run LLVM's FileCheck
function(FileCheck expected actual)
  CheckStatus(
    COMMAND ${FILECHECK} --comment=// ${expected} 
    EXPECTED_STATUS 0
    INPUT_FILE ${actual})
endfunction()

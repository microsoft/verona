if(NOT EXISTS ${PARSER})
  message(FATAL_ERROR " To run tests you must build the install target." ${PARSER})
endif()

find_package(Python3 COMPONENTS Interpreter)

if (NOT Python3_FOUND)
  message(WARNING " No python interpreter - Test Suite not compiled")
  return()
endif()

find_program(FILECHECK NAMES OutputCheck)

if(NOT FILECHECK)
  message(WARNING " Could not find OutputCheck - Test Suite not compiled")
  message(WARNING "   Run: pip install OutputCheck")
  return()
endif()

file(GLOB INPUT_FILES ${TEST_DIR}/*.verona)
foreach(INPUT_FILE ${INPUT_FILES})
  get_filename_component(TEST_NAME ${INPUT_FILE} NAME_WE)
  message(STATUS parse-${TEST_NAME})
  set(AST_DUMP_FILE ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}.output)
  execute_process(
    COMMAND ${PARSER} -g ${GRAMMAR} -a ${INPUT_FILE}
    OUTPUT_FILE ${AST_DUMP_FILE})

  if(NOT ${TEST_EXIT} EQUAL 0)
    message(FATAL_ERROR "Parser Exit " ${TEST_EXIT})
  endif()

  execute_process(
    COMMAND ${FILECHECK} --comment=// ${INPUT_FILE}
    INPUT_FILE ${AST_DUMP_FILE}
    RESULT_VARIABLE TEST_EXIT)

  file(READ ${AST_DUMP_FILE} AST_DUMP)
  message(STATUS ${AST_DUMP})

  if(NOT ${TEST_EXIT} EQUAL 0)
    message(FATAL_ERROR "Parse Failed")
  endif()
endforeach()

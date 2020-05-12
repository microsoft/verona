
file(GLOB TEST_FOLDERS ${CMAKE_CURRENT_LIST_DIR}/*)
foreach(TEST_FOLDER ${TEST_FOLDERS})
  if(NOT IS_DIRECTORY ${TEST_FOLDER})
    continue()
  endif()

  if(${TEST_FOLDER} MATCHES ".*/parse")
    file(GLOB PARSE_TESTS "${TEST_FOLDER}/ast-parse/*.verona")
    foreach(PARSE_TEST ${PARSE_TESTS})
      get_filename_component(TEST_NAME ${PARSE_TEST} NAME_WE)
      set(OUT_FILE ${TEST_FOLDER}/ast-parse/${TEST_NAME}/ast.txt)
      message(STATUS "Regenerating ${OUT_FILE}")
      execute_process(
        COMMAND ${PARSER} -a -g ${GRAMMAR} ${PARSE_TEST}
        OUTPUT_FILE ${OUT_FILE})
    endforeach()
  elseif(${TEST_FOLDER} MATCHES ".*/mlir")
    file(GLOB MLIR_TESTS "${TEST_FOLDER}/mlir-parse/*.verona" "${TEST_FOLDER}/mlir-parse/*.mlir")
    foreach(MLIR_TEST ${MLIR_TESTS})
      get_filename_component(TEST_NAME ${MLIR_TEST} NAME_WE)
      set(OUT_FILE ${TEST_FOLDER}/mlir-parse/${TEST_NAME}/out.mlir)
      message(STATUS "Regenerating ${OUT_FILE}")
      execute_process(
        COMMAND ${MLIRGEN} -g ${GRAMMAR} ${PARSE_TEST} -o -
        OUTPUT_FILE ${OUT_FILE})
    endforeach()
  else()
    execute_process(COMMAND ${PYTHON_EXECUTABLE}
      ${PROJECT_SOURCE_DIR}/utils/update_dump.py
      ${VERONAC}
      ${TEST_FOLDER})
  endif()
endforeach()

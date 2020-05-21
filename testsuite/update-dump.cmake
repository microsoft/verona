
file(GLOB TEST_FOLDERS ${CMAKE_CURRENT_LIST_DIR}/*)
foreach(TEST_FOLDER ${TEST_FOLDERS})
  if(NOT IS_DIRECTORY ${TEST_FOLDER})
    continue()
  endif()

  if(${TEST_FOLDER} MATCHES ".*/parse")
    execute_process(COMMAND ${PYTHON_EXECUTABLE}
      ${PROJECT_SOURCE_DIR}/utils/update_dump.py
      "${PARSER} -g ${GRAMMAR}"
      ${TEST_FOLDER})
  else()
    execute_process(COMMAND ${PYTHON_EXECUTABLE}
      ${PROJECT_SOURCE_DIR}/utils/update_dump.py
      ${VERONAC}
      ${TEST_FOLDER})
  endif()
endforeach()

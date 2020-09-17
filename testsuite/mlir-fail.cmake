include(${CMAKE_CURRENT_LIST_DIR}/common.cmake)

PrepareTest(VERONAC_FLAGS EXPECTED_DUMP ACTUAL_DUMP)

CheckStatus(
  COMMAND
    ${MLIRGEN} --verify-diagnostics --split-input-file
    ${TEST_FILE} -o -
  EXPECTED_STATUS 0)

include("${trieste_SOURCE_DIR}/cmake/testsuite.cmake")

# Use Trieste to test Verona bytecode tools (VC, VIRC, and VBCI).
testsuite(vbc)

add_test(
  NAME diagnostics-normalization
  COMMAND "${CMAKE_COMMAND}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tests/normalize_diagnostics_test.cmake")

add_test(
  NAME build-type-required
  COMMAND "${CMAKE_COMMAND}"
    "-DSOURCE_DIR:PATH=${PROJECT_SOURCE_DIR}"
    "-DCXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tests/build_type_test.cmake")

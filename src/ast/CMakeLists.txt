find_package(Threads REQUIRED)

add_library(verona-ast-lib
  ast.cc
  err.cc
  files.cc
  parser.cc
  sym.cc)

target_link_libraries(verona-ast-lib CLI11::CLI11)
target_link_libraries(verona-ast-lib cpp-peglib)
target_link_libraries(verona-ast-lib Threads::Threads)

add_executable(verona-ast main.cc cli.cc path.cc)
target_link_libraries(verona-ast verona-ast-lib)

install(TARGETS verona-ast RUNTIME DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/grammar.peg DESTINATION .)

enable_testing()

set(PARSER ${CMAKE_INSTALL_PREFIX}/verona-ast)
if(WIN32)
  set(PARSER ${PARSER}.exe)
endif()

add_test(NAME ast-parse COMMAND ${CMAKE_COMMAND}
  -DPARSER=${PARSER}
  -DGRAMMAR=${CMAKE_INSTALL_PREFIX}/grammar.peg
  -DTEST_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test
  -P ${CMAKE_CURRENT_SOURCE_DIR}/test/tests.cmake)

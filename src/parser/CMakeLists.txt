find_package(Threads REQUIRED)
find_package(CLI11 CONFIG REQUIRED)

add_library(verona-parser-lib
  anf.cc
  ast.cc
  dnf.cc
  escaping.cc
  lexer.cc
  lookup.cc
  parser.cc
  path.cc
  pretty.cc
  print.cc
  resolve.cc
  rewrite.cc
  source.cc)

add_executable(verona-parser main.cc)
target_link_libraries(verona-parser-lib CLI11::CLI11)
target_link_libraries(verona-parser-lib Threads::Threads)
target_link_libraries(verona-parser verona-parser-lib)

# Copy the stdlib into the place verona-mlir will find it.
add_custom_command(TARGET verona-parser
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} ARGS -E create_symlink ${CMAKE_SOURCE_DIR}/src/stdlib $<TARGET_FILE_DIR:verona-parser>/stdlib
  MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/src/stdlib)

install(TARGETS verona-parser RUNTIME DESTINATION .)

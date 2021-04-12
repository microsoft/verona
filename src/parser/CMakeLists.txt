find_package(Threads REQUIRED)

add_library(verona-parser-lib
  anf.cc
  ast.cc
  dnf.cc
  driver.cc
  escaping.cc
  infer.cc
  lexer.cc
  parser.cc
  path.cc
  pretty.cc
  print.cc
  resolve.cc
  rewrite.cc
  source.cc
  subtype.cc)

add_executable(verona-parser main.cc)
target_link_libraries(verona-parser-lib CLI11::CLI11)
target_link_libraries(verona-parser-lib Threads::Threads)
target_link_libraries(verona-parser verona-parser-lib)

install(TARGETS verona-parser RUNTIME DESTINATION .)

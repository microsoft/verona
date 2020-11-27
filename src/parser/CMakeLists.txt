add_library(verona-parser-lib lexer.cc parser.cc path.cc source.cc)
target_link_libraries(verona-parser-lib verona-ast-lib)

add_executable(verona-parser main.cc)
target_link_libraries(verona-parser
  verona-ast-lib
  verona-parser-lib)

install(TARGETS verona-parser RUNTIME DESTINATION .)

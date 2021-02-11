add_library(verona-typed-ast-lib conversion.cc print.cc)
target_link_libraries(verona-typed-ast-lib cpp-peglib)

add_executable(verona-typed-ast main.cc)
target_link_libraries(verona-typed-ast
  verona-ast-lib
  verona-typed-ast-lib)

install(TARGETS verona-typed-ast RUNTIME DESTINATION .)

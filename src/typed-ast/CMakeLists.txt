add_library(verona-typed-ast-lib conversion.cc print.cc)
target_link_libraries(verona-typed-ast-lib cpp-peglib)

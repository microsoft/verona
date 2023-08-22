add_executable(verona
  lang.cc
  lookup.cc
  main.cc
  parse.cc
  subtype.cc
  passes/modules.cc
  passes/structure.cc
  passes/memberconflict.cc
  passes/typenames.cc
  passes/typeview.cc
  passes/typefunc.cc
  passes/typealg.cc
  passes/typeflat.cc
  passes/typevalid.cc
  passes/codereuse.cc
  passes/conditionals.cc
  passes/reference.cc
  passes/reverseapp.cc
  passes/application.cc
  passes/assignlhs.cc
  passes/localvar.cc
  passes/assignment.cc
  passes/nlrcheck.cc
  passes/lambda.cc
  passes/autofields.cc
  passes/autorhs.cc
  passes/autocreate.cc
  passes/defaultargs.cc
  passes/partialapp.cc
  passes/traitisect.cc
  passes/anf.cc
  passes/defbeforeuse.cc
  passes/drop.cc
  passes/validtypeargs.cc
)

target_precompile_headers(verona
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/btype.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lang.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lookup.h
  ${CMAKE_CURRENT_SOURCE_DIR}/subtype.h
  ${CMAKE_CURRENT_SOURCE_DIR}/wf.h
)

target_link_libraries(verona
  Threads::Threads
  fmt::fmt
  trieste::trieste
)

add_test(NAME verona COMMAND verona test -f)

install(TARGETS verona RUNTIME DESTINATION verona)

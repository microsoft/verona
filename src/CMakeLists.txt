add_executable(verona
  lang.cc
  lookup.cc
  main.cc
  parse.cc
  subtype.cc
  passes/modules.cc
  passes/structure.cc
  passes/conditionals.cc
  passes/reference.cc
  passes/typenames.cc
  passes/typeview.cc
  passes/typefunc.cc
  passes/typealg.cc
  passes/typeflat.cc
  passes/typevalid.cc
  passes/typereference.cc
  passes/codereuse.cc
  passes/memberconflict.cc
  passes/resetimplicit.cc
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
  passes/typeinfer.cc
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

if (VERONA_SANITIZER)
  target_compile_options(verona PUBLIC
    -fsanitize=${VERONA_SANITIZER} -g -fno-omit-frame-pointer
  )
  target_link_options(verona PUBLIC
    -fsanitize=${VERONA_SANITIZER}
  )
endif()

add_test(NAME verona COMMAND verona test -f)

install(TARGETS verona RUNTIME DESTINATION verona)

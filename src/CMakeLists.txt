set (SRCS
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
  passes/namearity.cc
  passes/validtypeargs.cc
)

# The following code checks the manually specified set of source files is complete.
# CMake documentation does not recommend the use of GLOB_RECURSE as some platforms
# may not support CONFIGURE_DEPENDS correctly.  On platforms where it works correctly
# this should trap missing files.
list(SORT SRCS)
file(GLOB_RECURSE SRCS_GEN RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS *.cc)
list(SORT SRCS_GEN)
foreach (SRC SRC_GEN IN ZIP_LISTS SRCS SRCS_GEN)
  if (SRC STRGREATER SRC_GEN)
    message(FATAL_ERROR "SRCS are not up to date! Please update the srcs list. Missing file: ${SRC_GEN}")
  endif()
  if (SRC STRLESS SRC_GEN)
    message(FATAL_ERROR "SRCS are not up to date! Please update the srcs list. Extra file: ${SRC}")
  endif()
endforeach()

add_executable(verona ${SRCS})

target_link_libraries(verona
  Threads::Threads
  fmt::fmt
  trieste::trieste
)

add_test(NAME verona COMMAND verona test -f)

install(TARGETS verona RUNTIME DESTINATION verona)
install(DIRECTORY std DESTINATION verona)

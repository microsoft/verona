# Needed for pegmatite
add_definitions(-DUSE_RTTI=1)
set(CMAKE_CXX_STANDARD 17)

add_library(veronac-lib
  analysis.cc
  ast.cc
  codegen/builtins.cc
  codegen/codegen.cc
  codegen/descriptor.cc
  codegen/function.cc
  codegen/generator.cc
  codegen/reachability.cc
  codegen/selector.cc
  context.cc
  dataflow/liveness.cc
  elaboration.cc
  fixpoint.cc
  intern.cc
  ir/builder.cc
  ir/dominance.cc
  ir/ir.cc
  ir/point.cc
  ir/print.cc
  ir/variable_renaming.cc
  mapper.cc
  parser.cc
  polarize.cc
  printing.cc
  regionck/region_graph.cc
  resolution.cc
  type.cc
  typecheck/assertion.cc
  typecheck/capability_predicate.cc
  typecheck/constraint.cc
  typecheck/infer.cc
  typecheck/permission_check.cc
  typecheck/solver.cc
  typecheck/structural.cc
  typecheck/typecheck.cc
  typecheck/wf_types.cc
  ../../external/pegmatite/ast.cc
  ../../external/pegmatite/parser.cc
)

target_include_directories(veronac-lib PRIVATE ../../external/pegmatite)

target_link_libraries(veronac-lib fmt)
target_link_libraries(veronac-lib CLI11::CLI11)

add_executable(veronac main.cc)
target_link_libraries(veronac interpreter)
target_include_directories(veronac PRIVATE ../../external/pegmatite)
target_link_libraries(veronac veronac-lib)
install(TARGETS veronac RUNTIME DESTINATION .)

add_executable(veronac-sys main.cc)
target_compile_definitions(veronac-sys PRIVATE USE_SYSTEMATIC_TESTING)
target_include_directories(veronac-sys PRIVATE ../../external/pegmatite)
target_link_libraries(veronac-sys interpreter-sys)
target_link_libraries(veronac-sys veronac-lib)
install(TARGETS veronac-sys RUNTIME DESTINATION .)
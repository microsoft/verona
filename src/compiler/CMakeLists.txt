include(CheckCXXSymbolExists)

CHECK_CXX_SOURCE_COMPILES(
"
#include <filesystem>
int main()
{
  std::filesystem::path p;
}
"
 CXX_FILESYSTEM)

if (NOT CXX_FILESYSTEM)
  CHECK_CXX_SOURCE_COMPILES(
  "
  #include <experimental/filesystem>
  int main()
  {
    std::experimental::filesystem::path p;
  }
  "
  CXX_FILESYSTEM_EXPERIMENTAL)
endif()

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
  instantiation.cc
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
)

target_link_libraries(veronac-lib CLI11::CLI11)
target_link_libraries(veronac-lib fmt)
target_link_libraries(veronac-lib pegmatite-static)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  check_cxx_symbol_exists(_LIBCPP_VERSION ciso646 IS_LIBCXX)
  if (IS_LIBCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
    target_link_libraries(veronac-lib "c++experimental")
  elseif (IS_LIBCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(veronac-lib "c++fs")
  elseif(NOT IS_LIBCXX)
    target_link_libraries(veronac-lib "stdc++fs")
  endif()
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  target_link_libraries(veronac-lib "stdc++fs")
endif()

add_executable(veronac main.cc)
target_link_libraries(veronac interpreter)
target_link_libraries(veronac veronac-lib)
install(TARGETS veronac RUNTIME DESTINATION .)

add_executable(veronac-sys main.cc)
target_compile_definitions(veronac-sys PRIVATE USE_SYSTEMATIC_TESTING)
target_link_libraries(veronac-sys interpreter-sys)
target_link_libraries(veronac-sys veronac-lib)
install(TARGETS veronac-sys RUNTIME DESTINATION .)

if (CXX_FILESYSTEM)
  target_compile_definitions(veronac PRIVATE CXX_FILESYSTEM)
  target_compile_definitions(veronac-sys PRIVATE CXX_FILESYSTEM)
elseif (CXX_FILESYSTEM_EXPERIMENTAL)
  target_compile_definitions(veronac PRIVATE CXX_FILESYSTEM_EXPERIMENTAL)
  target_compile_definitions(veronac-sys PRIVATE CXX_FILESYSTEM_EXPERIMENTAL)
endif ()

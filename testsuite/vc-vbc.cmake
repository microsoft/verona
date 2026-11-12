include("${CMAKE_CURRENT_LIST_DIR}/cmake/should_run.cmake")

set(TESTSUITE_REGEX ".*\\.v$")
set(TESTSUITE_DEFINE vc_test_define)

function(vc_test_define test)
  get_filename_component(test_dir "${test}" DIRECTORY)
  get_filename_component(test_name "${test}" NAME_WE)
  get_filename_component(test_dir_name "${test_dir}" NAME)
  if(NOT test_name STREQUAL test_dir_name)
    return()
  endif()

  set(test_root "${test_dir}/${test_name}")
  set(compile_node "${test_root}/compile")
  set(run_node "${test_root}/run")

  testsuite_output_path(
    bytecode NODE "${compile_node}" FILE "${test_name}.vbc")
  testsuite_output_path(
    final_ast NODE "${compile_node}" FILE "${test_name}_final.trieste")

  verona_should_register_run(register_run "${test}")

  set(artifact_metadata)
  if(register_run)
    set(artifact_metadata ARTIFACTS "${test_name}.vbc")
  endif()

  testsuite_add_test(
    NAME "${compile_node}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${test_dir}"
    TIMEOUT 60
    GOLDENS exit_code.txt stderr.txt stdout.txt
    VALIDATOR "${CMAKE_CURRENT_SOURCE_DIR}/cmake/normalize_diagnostics.cmake"
    ${artifact_metadata}
    COMMAND
      "${CMAKE_INSTALL_PREFIX}/vc/vc"
      build .
      -b "${bytecode}"
      -o "${final_ast}")

  if(register_run)
    testsuite_add_test(
      NAME "${run_node}"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${test_root}"
      DEPENDS "${compile_node}"
      GOLDENS exit_code.txt stderr.txt stdout.txt
      COMMAND "${CMAKE_INSTALL_PREFIX}/vbci/vbci" "${bytecode}")
  endif()
endfunction()

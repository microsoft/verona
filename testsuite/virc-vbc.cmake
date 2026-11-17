include("${CMAKE_CURRENT_LIST_DIR}/cmake/should_run.cmake")

set(TESTSUITE_REGEX ".*\\.vir$")
set(TESTSUITE_DEFINE vir_test_define)

function(vir_test_define test)
  get_filename_component(test_dir "${test}" DIRECTORY)
  get_filename_component(test_file "${test}" NAME)
  get_filename_component(test_name "${test}" NAME_WE)
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

  set(compile_goldens exit_code.txt stderr.txt stdout.txt)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${compile_node}/${test_name}_final.trieste")
    list(APPEND compile_goldens "${test_name}_final.trieste")
  endif()

  testsuite_add_test(
    NAME "${compile_node}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${test_dir}"
    GOLDENS ${compile_goldens}
    ${artifact_metadata}
    COMMAND
      "${CMAKE_INSTALL_PREFIX}/virc/$<TARGET_FILE_NAME:virc>"
      build "${test_file}"
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

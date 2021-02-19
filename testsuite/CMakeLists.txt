enable_testing()


set(
  DISABLED_TESTS
  # Region-checking hasn't been ported yet
  veronac/features/compile-fail/bad-branch

  # The typechecker takes way too much time to solve this test
  veronac/ir/compile-pass/loop-complexity

  # These test randomly get stuck in a solver loop (microsoft/verona#77).
  # Disable it until we find out why.
  veronac/features/run-pass/when
  veronac/features/run-pass/loop
)

set (test_set)
function(subdirlist result curdir)
  file(GLOB children LIST_DIRECTORIES true CONFIGURE_DEPENDS RELATIVE ${curdir}  ${curdir}/*)
  set(dirlist "")
  foreach(child ${children})
    if(IS_DIRECTORY ${curdir}/${child})
      list(APPEND dirlist ${child})
    endif()
  endforeach()
  set(${result} ${dirlist} PARENT_SCOPE)
endfunction()

# Iterate each tool
subdirlist(TOOL_FOLDERS ${CMAKE_CURRENT_SOURCE_DIR})
foreach(TOOL ${TOOL_FOLDERS})
  set(TOOLCMD ${VERONA_LOCAL_DIST}/${TOOL}${CMAKE_EXECUTABLE_SUFFIX})
  set(TOOL_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/${TOOL})
  if (${TOOL} STREQUAL veronac)  # Remove this then we can use multiple tools when we are there
    file(GLOB_RECURSE tests CONFIGURE_DEPENDS  RELATIVE ${TOOL_FOLDER} ${TOOL_FOLDER}/*.verona)
    foreach(test ${tests})
      get_filename_component(test_name ${test} NAME_WE)
      get_filename_component(test_file ${test} NAME)
      get_filename_component(test_dir ${test} DIRECTORY)

      set (test_path ${TOOL}/${test_dir}/${test_name})

      list (FIND DISABLED_TESTS ${test_path} INDEX)
      if (NOT ${INDEX} EQUAL -1)
        continue()
      endif()

      set (output_dir ${CMAKE_CURRENT_BINARY_DIR}/${test_path}.out)
      set (test_output_cmd 
        ${CMAKE_COMMAND}
          -DTESTFILE=${test_file}
          -DWORKING_DIR=${TOOL_FOLDER}/${test_dir}
          -DTOOLCMD=${TOOLCMD}
          -DOUTPUT_DIR=${output_dir}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/run_command.cmake
      )

      # Add test that rebuilds the compiler output
      add_test(NAME ${test_path}.out
        COMMAND ${test_output_cmd}
      )

      # Add command that rebuilts the compiler output for updating golden files.
      add_custom_command(OUTPUT ${test_path}
        COMMAND ${test_output_cmd}
      )
      set_source_files_properties(${test_path} PROPERTIES SYMBOLIC "true")
      list(APPEND test_set ${test_path})

      set (golden_dir  ${CMAKE_CURRENT_SOURCE_DIR}/${test_path}.out )
      # Add tests for each golden file
      file (GLOB_RECURSE results CONFIGURE_DEPENDS RELATIVE ${golden_dir} ${golden_dir}/*)
      list(LENGTH results res_length)
      if(res_length EQUAL 0)
        message(WARNING "Test does not have tests of results. ${CMAKE_CURRENT_SOURCE_DIR}/${golden_dir}")
        # Add to generate golden output target
        add_custom_command(OUTPUT ${test_path}
          COMMAND
            ${CMAKE_COMMAND}
            -E make_directory
            ${golden_dir}
          APPEND
        )
        add_custom_command(OUTPUT ${test_path}
          COMMAND
            ${CMAKE_COMMAND}
            -E copy_if_different
            ${output_dir}/*
            ${golden_dir}/
          APPEND
        )
      else()
        foreach (result ${results})
          # Check each file is correct as a test target
          add_test (NAME ${test_path}.out/${result}
            COMMAND 
              ${CMAKE_COMMAND} -E compare_files --ignore-eol ${golden_dir}/${result} ${output_dir}/${result}
          )
          set_tests_properties(${test_path}.out/${result} PROPERTIES DEPENDS ${test_path}.out)

          # Override out of date files.
          add_custom_command(OUTPUT ${test_path}
            COMMAND
              ${CMAKE_COMMAND}
              -E copy_if_different
              ${output_dir}/${result}
              ${golden_dir}/${result}
            APPEND
          )
        endforeach()
      endif()
    endforeach()
  endif()
endforeach()

add_custom_target(update DEPENDS ${test_set})

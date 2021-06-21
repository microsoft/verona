enable_testing()

find_program(DIFF_TOOL NAMES
  diff)

if (DIFF_TOOL STREQUAL DIFF_TOOL-NOTFOUND)
  set(DIFF_TOOL "")
endif()

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
  set (test_set)
  set(TOOLCMD ${VERONA_LOCAL_DIST}/${TOOL}${CMAKE_EXECUTABLE_SUFFIX})
  set(TOOL_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/${TOOL})
  
  # Grab specific settings for this tool
  include(${CMAKE_CURRENT_LIST_DIR}/${TOOL}.cmake)

  # Use transform to support multiple extension for tests,
  # and find all the files with these extensions.
  list(TRANSFORM TEST_EXTENSION PREPEND ${TOOL_FOLDER}/*.)
  file(GLOB_RECURSE tests CONFIGURE_DEPENDS  RELATIVE ${TOOL_FOLDER} ${TEST_EXTENSION})

  foreach(test ${tests})
    get_filename_component(test_name ${test} NAME_WE)
    get_filename_component(test_file ${test} NAME)
    get_filename_component(test_dir ${test} DIRECTORY)

    if (test_dir STREQUAL "")
      set (test_path ${TOOL}/${test_name})
    else()
      set (test_path ${TOOL}/${test_dir}/${test_name})
    endif()

    list (FIND DISABLED_TESTS ${test_path} INDEX)
    if (NOT ${INDEX} EQUAL -1)
      message("Test currently disabled ${test_path}")
      continue()
    endif()

    # Create command to create the output for this test.
    set (output_dir ${CMAKE_CURRENT_BINARY_DIR}/${test_path}.out)
    set (test_output_cmd 
      ${CMAKE_COMMAND}
        -DTESTFILE=${test_file}
        -DWORKING_DIR=${TOOL_FOLDER}/${test_dir}
        -DTOOLCMD=${TOOLCMD}
        -DTOOLNAME=${TOOL}
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

    # Add output comparison for each golden / output file
    set (golden_dir  ${CMAKE_CURRENT_SOURCE_DIR}/${test_path}.out )
    file (GLOB_RECURSE results CONFIGURE_DEPENDS RELATIVE ${golden_dir} ${golden_dir}/*)
    # Check if there are any files to compare for this test.
    list(LENGTH results res_length)
    if(res_length EQUAL 0)
      message(WARNING "Test does not have results directory: ${golden_dir}")
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
            ${CMAKE_COMMAND}
              -Doriginal_file=${golden_dir}/${result} 
              -Dnew_file=${output_dir}/${result}
              -Ddiff_tool=${DIFF_TOOL}
              -P ${CMAKE_CURRENT_SOURCE_DIR}/compare.cmake
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
      # All tests require an error_code.
      add_custom_command(OUTPUT ${test_path}
        COMMAND
          ${CMAKE_COMMAND}
          -E copy_if_different
          ${output_dir}/exit_code.txt
          ${golden_dir}/exit_code.txt
        APPEND
      )

    endif()
  endforeach()
  add_custom_command(OUTPUT ${TOOL}
    DEPENDS ${test_set})
endforeach()

add_custom_target(update-dump DEPENDS ${TOOL_FOLDERS})

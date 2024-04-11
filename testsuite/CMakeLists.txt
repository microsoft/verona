enable_testing()

find_program(DIFF_TOOL NAMES
  diff)

if (DIFF_TOOL STREQUAL DIFF_TOOL-NOTFOUND)
  set(DIFF_TOOL "")
endif()

set(DISABLED_TESTS)

set(GOLDEN_DIR_SUFFIX "_out")

set(VERONA_LOCAL_DIST ${CMAKE_BINARY_DIR}/dist)
message("Verona local dist: ${VERONA_LOCAL_DIST}")

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
set(UPDATE_DUMPS_TARGETS)
foreach(TOOL ${TOOL_FOLDERS})
  if (TOOL MATCHES "deprecated")
    continue()
  endif()
  set (test_set)
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
    set (output_dir ${CMAKE_CURRENT_BINARY_DIR}/${test_path}${GOLDEN_DIR_SUFFIX})
    set (test_output_cmd 
      ${CMAKE_COMMAND}
        -DTESTFILE=${test_file}
        -DWORKING_DIR=${TOOL_FOLDER}/${test_dir}
        -DTOOLNAME=${TOOL}
        -DCMAKE_EXECUTABLE_SUFFIX=${CMAKE_EXECUTABLE_SUFFIX}
        -DVERONA_LOCAL_DIST=${VERONA_LOCAL_DIST}
        -DOUTPUT_DIR=${output_dir}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_command.cmake
    )

    # Add test that rebuilds the compiler output
    add_test(NAME ${test_path}${GOLDEN_DIR_SUFFIX}
      COMMAND ${test_output_cmd}
    )

    # Add command that rebuilts the compiler output for updating golden files.
    add_custom_command(OUTPUT ${test_path}
      COMMAND ${test_output_cmd}
    )
    set_source_files_properties(${test_path} PROPERTIES SYMBOLIC "true")
    list(APPEND test_set ${test_path})

    # Make json for debugging.
    toolinvoke(launch_json_args ${VERONA_LOCAL_DIST} ${test_file} ${output_dir})
    list(POP_FRONT launch_json_args launch_json_prog)
    # Convert to a json format list.
    string(REPLACE "\"" "\\\"" launch_json_args "${launch_json_args}")
    string(REPLACE ";" "\", \"" launch_json_args "${launch_json_args}")
    list(APPEND LAUNCH_JSON
"    {
      \"name\": \"${test_path}\",
      \"type\": \"cppdbg\",
      \"request\": \"launch\",
      \"program\": \"${launch_json_prog}\",
      \"args\": [\"${launch_json_args}\"],
      \"stopAtEntry\": false,
      \"cwd\": \"${TOOL_FOLDER}/${test_dir}\",
    },")

    # Add output comparison for each golden / output file
    set (golden_dir  ${CMAKE_CURRENT_SOURCE_DIR}/${test_path}${GOLDEN_DIR_SUFFIX} )
    file (GLOB_RECURSE results CONFIGURE_DEPENDS RELATIVE ${golden_dir} ${golden_dir}/*)
    # Check if there are any files to compare for this test.
    list(LENGTH results res_length)
    if(res_length EQUAL 0)
      message(WARNING "Test does not have results directory: ${golden_dir}\nRun `update-dump` to generate golden files.")
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
        add_test (NAME ${test_path}${GOLDEN_DIR_SUFFIX}/${result}
          COMMAND 
            ${CMAKE_COMMAND}
              -Doriginal_file=${golden_dir}/${result} 
              -Dnew_file=${output_dir}/${result}
              -Ddiff_tool=${DIFF_TOOL}
              -P ${CMAKE_CURRENT_SOURCE_DIR}/compare.cmake
        )
        set_tests_properties(${test_path}${GOLDEN_DIR_SUFFIX}/${result} PROPERTIES DEPENDS ${test_path}${GOLDEN_DIR_SUFFIX})

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
  add_custom_target("update-dump-${TOOL}" DEPENDS ${test_set})
  list(APPEND UPDATE_DUMPS_TARGETS "update-dump-${TOOL}")
endforeach()

string(REPLACE ";" "\n" LAUNCH_JSON "${LAUNCH_JSON}")

if (VERONA_GENERATE_LAUNCH_JSON)
  file(WRITE ${CMAKE_SOURCE_DIR}/.vscode/launch.json
"{
  \"version\": \"0.2.0\",
  \"configurations\": [
    ${LAUNCH_JSON}
  ]
}")
endif()


add_custom_target(update-dump DEPENDS ${UPDATE_DUMPS_TARGETS})

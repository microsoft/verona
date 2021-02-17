enable_testing()

if(${CMAKE_VERSION} VERSION_LESS "3.12.0")
  message(WARNING "CMake version too old - Update CMake to at least 3.12.0 for test suite.")
  if (VERONA_CI_BUILD)
    message(FATAL_ERROR "Tests must be built in CI!")
  endif()
  return()
endif ()

find_package(Python3 COMPONENTS Interpreter)

if(NOT Python3_FOUND)
  message(WARNING " No python3 interpreter - Test Suite not compiled")
  if(VERONA_CI_BUILD)
    message(FATAL_ERROR "Tests must be built in CI!")
  endif()
  return()
endif()

find_program(FILECHECK
  NAMES OutputCheck)

if(NOT FILECHECK)
  message(WARNING " Could not find OutputCheck - Test Suite not compiled")
  message(WARNING "   Run: pip install OutputCheck")
  if (VERONA_CI_BUILD)
    message(FATAL_ERROR "Tests must be built in CI!")
  endif()
  return()
endif()

set(VERONAC ${VERONA_LOCAL_DIST}/veronac${CMAKE_EXECUTABLE_SUFFIX})
set(VERONAI ${VERONA_LOCAL_DIST}/interpreter${CMAKE_EXECUTABLE_SUFFIX})
set(VERONA_AST ${VERONA_LOCAL_DIST}/verona-ast${CMAKE_EXECUTABLE_SUFFIX})
set(VERONA_TYPED_AST ${VERONA_LOCAL_DIST}/verona-typed-ast${CMAKE_EXECUTABLE_SUFFIX})
set(VERONA_PARSER ${VERONA_LOCAL_DIST}/verona-parser${CMAKE_EXECUTABLE_SUFFIX})
set(VERONA_MLIR ${VERONA_LOCAL_DIST}/verona-mlir${CMAKE_EXECUTABLE_SUFFIX})

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

function(add_tests mode dir)
  set(testdir ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/${mode})
  file(GLOB filenames CONFIGURE_DEPENDS RELATIVE ${testdir} ${testdir}/*.verona ${testdir}/*.mlir)
  foreach(filename ${filenames})
    get_filename_component(stem ${filename} NAME_WE)
    set(testname ${dir}/${mode}/${stem})

    file(TO_NATIVE_PATH ${testdir}/${filename} testfilename)

    add_test(NAME ${testname} COMMAND ${CMAKE_COMMAND}
      -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}
      -DVERONAC=${VERONAC}
      -DINTERPRETER=${VERONAI}
      -DVERONA_AST=${VERONA_AST}
      -DVERONA_PARSER=${VERONA_PARSER}
      -DVERONA_TYPED_AST=${VERONA_TYPED_AST}
      -DVERONA_MLIR=${VERONA_MLIR}
      -DGRAMMAR=${VERONA_LOCAL_DIST}/grammar.peg
      -DFILECHECK=${FILECHECK}
      -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
      -DCHECK_DUMP_PY=${PROJECT_SOURCE_DIR}/utils/check_dump.py
      -DTEST_NAME=${testname}
      -DTEST_FILE=${testfilename}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/${mode}.cmake)
  endforeach()
endfunction()

subdirlist(TEST_FOLDERS ${CMAKE_CURRENT_SOURCE_DIR})
foreach(TEST_FOLDER ${TEST_FOLDERS})
  add_tests(compile-pass ${TEST_FOLDER})
  add_tests(compile-fail ${TEST_FOLDER})
  add_tests(run-pass ${TEST_FOLDER})
  add_tests(ast-parse ${TEST_FOLDER})
  add_tests(parser ${TEST_FOLDER})
  add_tests(mlir-parse ${TEST_FOLDER})
  add_tests(mlir-fail ${TEST_FOLDER})
endforeach()


add_custom_target(build_test_output)

add_custom_command(
  OUTPUT build_golden_output
  COMMAND echo ""
  )
set_source_files_properties(build_golden_output PROPERTIES SYMBOLIC "true")
add_custom_target(update DEPENDS build_golden_output)


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

      # Build dump for this run
      set (test_path ${TOOL}/${test_dir}/${test_name})
      set (output_dir ${CMAKE_CURRENT_BINARY_DIR}/${test_path}.out)
      add_test(NAME ${test_path}
        COMMAND 
          ${CMAKE_COMMAND}
            -DTESTFILE=${test_file}
            -DWORKING_DIR=${TOOL_FOLDER}/${test_dir}
            -DTOOLCMD=${TOOLCMD}
            -DOUTPUT_DIR=${output_dir}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_command.cmake
      )

      add_custom_command(OUTPUT build_golden_output
      COMMAND
        ${CMAKE_COMMAND}
        -DTESTFILE=${test_file}
        -DWORKING_DIR=${TOOL_FOLDER}/${test_dir}
        -DTOOLCMD=${TOOLCMD}
        -DOUTPUT_DIR=${golden_dir}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_command.cmake
      APPEND
      )

      set (golden_dir  ${CMAKE_CURRENT_SOURCE_DIR}/${test_path}.out )
      # Add tests for each golden file
      file (GLOB_RECURSE results CONFIGURE_DEPENDS RELATIVE ${golden_dir} ${golden_dir}/*)
      list(LENGTH results res_length)
      if(res_length EQUAL 0)
        message(WARNING "Test does not have tests of results. ${CMAKE_CURRENT_SOURCE_DIR}/${golden_dir}")
        # Add to generate golden output target
        # add_custom_command(OUTPUT build_golden_output
        #   COMMAND
        #     ${CMAKE_COMMAND}
        #     -DTESTFILE=${test_file}
        #     -DWORKING_DIR=${TOOL_FOLDER}/${test_dir}
        #     -DTOOLCMD=${TOOLCMD}
        #     -DOUTPUT_DIR=${golden_dir}.test
        #     -P ${CMAKE_CURRENT_SOURCE_DIR}/run_command.cmake
        #   APPEND
        # )
      else()
        foreach (result ${results})
          add_test (NAME ${test_path}/${result}
            COMMAND 
              ${CMAKE_COMMAND} -E compare_files --ignore-eol ${golden_dir}/${result} ${output_dir}/${result}
          )
          set_tests_properties(${test_path}/${result} PROPERTIES DEPENDS ${test_path})
        endforeach()
      endif()

      # If no stderr.txt output file, then use Filecheck

    endforeach()
  endif()
endforeach()


# set_tests_properties(
#   # Region-checking hasn't been ported yet
#   veronac/features/compile-fail/bad-branch

#   # The typechecker takes way too much time to solve this test
#   veronac/ir/compile-pass/loop-complexity

#   # These test randomly get stuck in a solver loop (microsoft/verona#77).
#   # Disable it until we find out why.
#   veronac/features/run-pass/when
#   veronac/features/run-pass/loop

#   PROPERTIES DISABLED true)

add_custom_target(update-dump COMMAND ${CMAKE_COMMAND}
  -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}
  -DVERONAC=${VERONAC}
  -DVERONA_AST=${VERONA_AST}
  -DVERONA_PARSER=${VERONA_PARSER}
  -DVERONA_TYPED_AST=${VERONA_TYPED_AST}
  -DVERONA_MLIR=${VERONA_MLIR}
  -DGRAMMAR=${VERONA_LOCAL_DIST}/grammar.peg
  -DPROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}
  -P ${CMAKE_CURRENT_SOURCE_DIR}/update-dump.cmake)

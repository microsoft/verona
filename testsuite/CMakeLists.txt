enable_testing()

if (${CMAKE_VERSION} VERSION_LESS "3.12.0")
  message(WARNING "CMake version too old - Update CMake to at least 3.12.0 for test suite.")
  if (VERONA_CI_BUILD)
    message(FATAL_ERROR "Tests must be built in CI!")
  endif()
  return()
endif ()

find_package(Python3 COMPONENTS Interpreter)

if (NOT Python3_FOUND)
  message(WARNING " No python interpreter - Test Suite not compiled")
  if (VERONA_CI_BUILD)
    message(FATAL_ERROR "Tests must be built in CI!")
  endif()
  return()
endif()

find_program(FILECHECK
  NAMES OutputCheck
)

if(NOT FILECHECK)
  message(WARNING " Could not find OutputCheck - Test Suite not compiled")
  message(WARNING "   Run: pip install OutputCheck")
  if (VERONA_CI_BUILD)
    message(FATAL_ERROR "Tests must be built in CI!")
  endif()
  return()
endif()

if (WIN32)
  set(VERONAC ${CMAKE_INSTALL_PREFIX}/veronac.exe)
  set(VERONAI ${CMAKE_INSTALL_PREFIX}/interpreter.exe)
else ()
  set(VERONAC ${CMAKE_INSTALL_PREFIX}/veronac)
  set(VERONAI ${CMAKE_INSTALL_PREFIX}/interpreter)
endif()

function(subdirlist result curdir)
  file(GLOB children LIST_DIRECTORIES true RELATIVE ${curdir}  ${curdir}/*)
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
  file(GLOB filenames RELATIVE ${testdir} ${testdir}/*.verona)
  foreach(filename ${filenames})
    get_filename_component(stem ${filename} NAME_WE)
    set(testname ${dir}/${mode}/${stem})

    file(TO_NATIVE_PATH ${testdir}/${filename} testfilename)

    add_test(NAME ${testname} COMMAND ${CMAKE_COMMAND}
      -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}
      -DVERONAC=${VERONAC}
      -DINTERPRETER=${VERONAI}
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
endforeach()

set_tests_properties(
  # Region-checking hasn't been ported yet
  features/compile-fail/bad-branch

  # The typechecker takes way too much time to solve this test
  ir/compile-pass/loop-complexity

  # This test randomly times out (microsoft/verona#77).
  # Disable it until we find out why.
  features/run-pass/when

  PROPERTIES DISABLED true)

add_custom_target(update-dump
  COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/utils/update_dump.py
    ${VERONAC}
    ${CMAKE_CURRENT_SOURCE_DIR})

# Arguments for testing vbcc samples
macro(toolinvoke ARGS testfile outputdir)
  get_filename_component(test_name ${testfile} NAME_WE)
  set(${ARGS} build ${testfile} -o ${outputdir}/${test_name}_final.trieste --dump_passes ${outputdir}) 
endmacro()

# Regular expression to match test files
# This regex matches files with the .infix extension
set(TESTSUITE_REGEX ".*\\.infix")

set(TESTSUITE_EXE "$<TARGET_FILE:infixlang>")

function (test_output_dir out test)
  # Use get_filename_component to remove the file extension and keep the directory structure
  get_filename_component(test_dir ${test} DIRECTORY)
  get_filename_component(test_name ${test} NAME_WE)
  # Create the output directory relative to the test directory
  set(${out} "generated/${test_dir}/${test_name}/" PARENT_SCOPE)
endfunction()
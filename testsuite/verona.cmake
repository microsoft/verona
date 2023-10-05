# Arguments for testing verona-parser
macro(toolinvoke ARGS local_dist testfile outputdir)
  set(${ARGS} "${local_dist}/verona/verona${CMAKE_EXECUTABLE_SUFFIX}" build ${testfile} --wf-check -o ${outputdir}/ast)
endmacro()

# Set extension for verona tests
set(TEST_EXTENSION "verona")

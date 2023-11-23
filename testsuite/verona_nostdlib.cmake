# Arguments for testing verona-parser
macro(toolinvoke ARGS local_dist testfile outputdir)
  set(${ARGS} "${local_dist}/verona/verona${CMAKE_EXECUTABLE_SUFFIX}" build ${testfile} -w -o ${outputdir}/ast --dump_passes ${outputdir} --no-std -p defaultargs)
endmacro()

# Set extension for verona tests
set(TEST_EXTENSION "verona")

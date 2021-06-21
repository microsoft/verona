# Arguments for testing Verona-interop
macro(toolargs ARGS testfile outputdir)
  set(${ARGS} -config ${testfile} )
endmacro()

# Set extension for verona-interop tests
set(TEST_EXTENSION "cfg")

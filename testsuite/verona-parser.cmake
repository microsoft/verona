# Arguments for testing verona-parser
macro(toolargs ARGS testfile outputdir)
  set(${ARGS} ${testfile} --ast --validate --anf)
endmacro()

# Set extension for verona tests
set(TEST_EXTENSION "verona")

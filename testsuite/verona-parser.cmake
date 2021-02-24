# Arguments for testing Veronac
macro(toolargs ARGS testfile outputdir)
  set(${ARGS} ${testfile} --ast)
endmacro()

# Set extension for verona tests
set(TEST_EXTENSION "verona")

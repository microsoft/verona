# Arguments for testing Veronac
macro(toolargs ARGS testfile outputdir)
  set(${ARGS} ${testfile} --dump-path ${outputdir} --disable-colors --run)
endmacro()

# Set extension for verona tests
set(TEST_EXTENSION "verona")

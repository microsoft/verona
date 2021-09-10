# Arguments for testing verona-mlir
macro(toolargs ARGS testfile outputdir)
set(${ARGS} -config ${testfile})
endmacro()

# Set extension for verona-mlir tests
set(TEST_EXTENSION cfg)


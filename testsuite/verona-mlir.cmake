# Arguments for testing verona-mlir
macro(toolargs ARGS testfile outputdir)
set(${ARGS} ${testfile} -o -)
endmacro()

# Set extension for verona-mlir tests
set(TEST_EXTENSION verona mlir)


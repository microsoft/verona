# This is used to run a command that can fail.
# Dumping all the output and error code into a file
# Also handles timeouts

make_directory(${OUTPUT_DIR})
# Run command
execute_process(
    COMMAND ${TOOLCMD} ${TESTFILE} #--dump-path ${OUTPUT_DIR}
    WORKING_DIRECTORY ${WORKING_DIR}
    OUTPUT_FILE ${OUTPUT_DIR}/stdout.txt
    ERROR_FILE ${OUTPUT_DIR}/stderr.txt
    TIMEOUT 10   # Timeout at 10 seconds, may need to increase this.
    RESULT_VARIABLE status
)

# Push error code into dump
file(WRITE ${OUTPUT_DIR}/error_code.txt ${status})

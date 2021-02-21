# This is used to run a command that can fail.
# Dumping all the output and error code into a file
# Also handles timeouts

file(REMOVE_RECURSE ${OUTPUT_DIR})

make_directory(${OUTPUT_DIR})

include(${CMAKE_CURRENT_LIST_DIR}/${TOOLNAME}.cmake)

toolargs(TOOLARGS ${TESTFILE} ${OUTPUT_DIR})

# Run command
execute_process(
    COMMAND ${TOOLCMD} ${TOOLARGS}#${TESTFILE} --dump-path ${OUTPUT_DIR} --run
    WORKING_DIRECTORY ${WORKING_DIR}
    OUTPUT_FILE ${OUTPUT_DIR}/stdout.txt
    ERROR_FILE ${OUTPUT_DIR}/stderr.txt
    TIMEOUT 20   # Timeout at 60 seconds, may need to increase this.
    RESULT_VARIABLE status
)

# Push error code into dump
file(WRITE ${OUTPUT_DIR}/error_code.txt ${status})

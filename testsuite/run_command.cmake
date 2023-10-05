# This is used to run a command that can fail.
# Dumping all the output and error code into a file
# Also handles timeouts

file(REMOVE_RECURSE ${OUTPUT_DIR})

make_directory(${OUTPUT_DIR})

include(${CMAKE_CURRENT_LIST_DIR}/${TOOLNAME}.cmake)

message("Verona Local dist: ${VERONA_LOCAL_DIST}")
toolinvoke(TOOLINVOKE ${VERONA_LOCAL_DIST} ${TESTFILE} ${OUTPUT_DIR})

list(JOIN TOOLINVOKE " " TOOLINVOKE_SEP)
message ("Running")
message ("   ${TOOLINVOKE_SEP}")
message ("in working directory")
message ("   ${WORKING_DIR}")
message ("output sent to")
message ("   ${OUTPUT_DIR}")

# Run command
execute_process(
    COMMAND ${TOOLINVOKE}
    WORKING_DIRECTORY ${WORKING_DIR}
    OUTPUT_FILE ${OUTPUT_DIR}/stdout.txt
    ERROR_FILE ${OUTPUT_DIR}/stderr.txt
    TIMEOUT 20   # Timeout at 20 seconds, may need to increase this.
    RESULT_VARIABLE status
)

# Push exit code into dump and make sure both stdout and stderr exist
file(WRITE ${OUTPUT_DIR}/exit_code.txt ${status})
file(TOUCH ${OUTPUT_DIR}/stdout.txt)
file(TOUCH ${OUTPUT_DIR}/stderr.txt)

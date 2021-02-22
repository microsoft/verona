# This is used to compare to files using CMake.
# It improves on the default behaviour to print the
# the files if there is a difference.

execute_process(
    COMMAND ${CMAKE_COMMAND} -E compare_files --ignore-eol ${original_file} ${new_file}
    RESULT_VARIABLE status
)

if (${status} EQUAL 1)
    message ("Compare ${original_file} with ${new_file}")
    if (diff_tool STREQUAL "")
        file(READ ${original_file} original_text)
        file(READ ${new_file} new_text)
        message("--Original File-----------------------------------------------------------------")
        if (NOT original_text STREQUAL "")
            message("${original_text}")
        endif()
        message("--------------------------------------------------------------------------------")
        message("  ")
        message("--New File----------------------------------------------------------------------")
        if (NOT new_text STREQUAL "")
            message(${new_text})
        endif()
        message("--------------------------------------------------------------------------------")
    else ()
        execute_process(
            COMMAND ${diff_tool} ${original_file} ${new_file}
        )
    endif ()
    message(FATAL_ERROR "Files differ!")
endif ()
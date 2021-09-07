include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(external/pegmatite EXCLUDE_FROM_ALL)
add_subdirectory(compiler)
add_subdirectory(interpreter)

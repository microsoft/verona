include_directories(.)

option(RT_TESTS "Including unit tests for the runtime" OFF)

set(CMAKE_CXX_STANDARD 17)

# Main target does not test runtime, build subdirectory for that
if (RT_TESTS)
else()
  set (VERONA_RT_ONLY_HEADER_LIBRARY ON)
endif()
add_subdirectory(rt)


# Should go to stricter warnings 
#   warnings_high()
# But for now, just add some to stop the Windows build breaking
if(MSVC)
    add_compile_options(/WX)
endif()
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wmismatched-tags)
endif()


add_subdirectory(compiler)
add_subdirectory(interpreter)
add_subdirectory(stdlib)

clangformat_targets()

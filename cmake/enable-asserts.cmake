# Copied from LLVM, handling release builds with assertions isn't trivial as
# CMake defines -DNDEBUG automatically and we have to clear it.
macro(enable_asserts)
  # MSVC doesn't like _DEBUG on release builds.
  if( NOT MSVC )
    message(STATUS "Enabling asserts")
    add_definitions( -D_DEBUG )
  endif()
  # On non-Debug builds cmake automatically defines NDEBUG, so we
  # explicitly undefine it:
  if( NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG" )
    message(STATUS "Removing NDEBUG from existing flags to enable asserts")
    # NOTE: use `add_compile_options` rather than `add_definitions` since
    # `add_definitions` does not support generator expressions.
    add_compile_options($<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-UNDEBUG>)

    # Also remove /D NDEBUG to avoid MSVC warnings about conflicting defines.
    foreach (flags_var_to_scrub
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS_MINSIZEREL
        CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_RELWITHDEBINFO
        CMAKE_C_FLAGS_MINSIZEREL)
      string (REGEX REPLACE "(^| )[/-]D *NDEBUG($| )" " "
        "${flags_var_to_scrub}" "${${flags_var_to_scrub}}")
    endforeach()
  endif()
endmacro()

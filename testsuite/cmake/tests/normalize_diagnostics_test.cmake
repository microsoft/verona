include("${CMAKE_CURRENT_LIST_DIR}/../normalize_diagnostics.cmake")

set(expected "-- _builtin/ffi/callback.v:13:39")
foreach(prefix
    "../../../../build/dist/vc"
    "../../../../build/debug/dist/vc"
    "../../../../build/release/dist/vc"
    "/tmp/worktree with spaces/build/debug/dist/vc"
    "C:/worktree/build/release/dist/vc")
  set(input "Errors:\n-- ${prefix}/_builtin/ffi/callback.v:13:39\nmessage\n")
  verona_normalize_diagnostics(actual "${input}")
  if(NOT actual STREQUAL "Errors:\n${expected}\nmessage\n")
    message(FATAL_ERROR "Builtin diagnostic normalization failed: ${actual}")
  endif()
endforeach()

set(unchanged
  "-- _builtin/ffi/callback.v:14:40\n-- ../user.v:2:3\n"
  "message mentions build/debug/dist/vc/_builtin/ffi/callback.v:13:39\n"
  "-- ../build/debug/dist/vc/_builtin/not-a-location\n"
  " --> _builtin/ffi/exit.v:8:3\n")
string(JOIN "" unchanged ${unchanged})
verona_normalize_diagnostics(actual "${unchanged}")
if(NOT actual STREQUAL unchanged)
  message(FATAL_ERROR "Normalization changed stable diagnostic text: ${actual}")
endif()

verona_normalize_diagnostics(actual
  "-- ../build/debug/dist/vc/_builtin/a.v:1:2\n-- ../build/release/dist/vc/_builtin/b.v:3:4\n")
if(NOT actual STREQUAL "-- _builtin/a.v:1:2\n-- _builtin/b.v:3:4\n")
  message(FATAL_ERROR "Consecutive source locations were not preserved: ${actual}")
endif()

function(verona_normalize_diagnostics out text)
  # Only builtin source-location headers contain installation-dependent paths.
  string(REGEX REPLACE
    "(^|\n)-- [^\r\n]*[/\\\\]_builtin/([^:\r\n]+:[0-9]+:[0-9]+)"
    "\\1-- _builtin/\\2" normalized "${text}")
  set(${out} "${normalized}" PARENT_SCOPE)
endfunction()

if(DEFINED OUTPUT_DIR)
  foreach(stream stdout stderr)
    set(path "${OUTPUT_DIR}/${stream}.txt")
    file(READ "${path}" text)
    verona_normalize_diagnostics(normalized "${text}")
    file(WRITE "${path}" "${normalized}")
  endforeach()
endif()

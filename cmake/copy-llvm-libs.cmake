list(GET SOURCES 0 base)
list(SUBLIST SOURCES 1 -1 SOURCES)

message(STATUS "Copying ${base}...")
file(COPY ${base} DESTINATION ${DESTINATION})

foreach(name ${SOURCES})
  message(STATUS "Copying ${name}...")
  file(COPY
    ${name} 
    DESTINATION ${DESTINATION}
    FILES_MATCHING
    PATTERN "*.a"
    PATTERN "*.lib"
    PATTERN "*.dylib"
    PATTERN "*.so"
    PATTERN "*.dll"
    PATTERN "MLIRTargets-*.cmake"
    PATTERN "LLVMExports-*.cmake")
endforeach()
message(STATUS "Done.")

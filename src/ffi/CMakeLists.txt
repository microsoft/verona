find_package(Threads REQUIRED)

link_directories(${LLVM_BUILD_LIBRARY_DIR})

add_library(verona-ffi-compiler
  compiler.cc
  )

add_executable(verona-ffi main.cc compiler.cc)
target_link_libraries(verona-ffi verona-ffi-compiler)

install(TARGETS verona-ffi RUNTIME DESTINATION .)

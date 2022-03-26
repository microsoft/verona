find_package(Threads REQUIRED)

add_library(verona-path-lib
  path.cc
  )

target_link_libraries(verona-path-lib
  Threads::Threads
  )

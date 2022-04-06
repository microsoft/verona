find_package(Threads REQUIRED)

add_executable(langkit-sample
  infer.cc
  lang.cc
  lookup.cc
  main.cc
  parse.cc
  )

# target_link_libraries(langkit-lib
#   Threads::Threads
#   )

target_link_libraries(langkit-sample
  Threads::Threads
  CLI11::CLI11
  verona-path-lib
  # langkit-lib
  )

install(TARGETS langkit-sample RUNTIME DESTINATION .)

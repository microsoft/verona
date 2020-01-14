find_package(Threads REQUIRED)

add_library(interpreter
  bytecode.cc
  interpreter.cc
  object.cc
  value.cc
  vm.cc
)

target_link_libraries(interpreter CLI11::CLI11)
target_link_libraries(interpreter fmt)
target_link_libraries(interpreter Threads::Threads)
target_link_libraries(interpreter verona_rt)

add_library(interpreter-sys
  bytecode.cc
  interpreter.cc
  object.cc
  value.cc
  vm.cc
)

target_link_libraries(interpreter-sys CLI11::CLI11)
target_link_libraries(interpreter-sys fmt)
target_link_libraries(interpreter-sys Threads::Threads)
target_link_libraries(interpreter-sys verona_rt)
target_compile_definitions(interpreter-sys PRIVATE USE_SYSTEMATIC_TESTING)


add_executable(interpreter-bin main.cc)
set_target_properties(interpreter-bin PROPERTIES OUTPUT_NAME interpreter)
target_link_libraries(interpreter-bin interpreter)

add_executable(interpreter-sys-bin main.cc)
set_target_properties(interpreter-sys-bin PROPERTIES OUTPUT_NAME interpreter-sys)
target_link_libraries(interpreter-sys-bin interpreter-sys)


install(TARGETS interpreter-bin RUNTIME DESTINATION .)
install(TARGETS interpreter-sys-bin RUNTIME DESTINATION .)

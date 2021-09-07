find_package(Threads REQUIRED)

# std::iterator is used by MLIR/LLVM
# This is deprecated and causes warnings in MSVC.
# Once fixed in LLVM, the following three lines can be removed.
if (MSVC)
  add_compile_definitions(_SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING)
endif ()

# LLVM requires CMake 3.13+ and this policy is for 3.12-, so we force new to
# avoid the CMake warnings on LLVM CMake files that we can't change
cmake_policy(SET CMP0077 NEW)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Disable RTTI and EH
if (MSVC)
  add_compile_options(/EH-)
  add_compile_options(/GR-)
else()
  add_compile_options(-fno-rtti)
  add_compile_options(-fno-exceptions)
endif()


### Verona Opt (MLIR-to-MLIR/LLVM pipeline)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRExecutionEngine
        MLIROptLib
        MLIRTargetLLVMIRExport
        verona-parser-lib
        )
add_executable(verona-mlir
  consumer.cc
  driver.cc
  error.cc
  generator.cc
  verona-mlir.cc)

get_target_property(STDLIB_TMP_PATH verona-mlir BINARY_DIR)
message(STATUS "Copying stdlib to ${STDLIB_TMP_PATH}")

file(CREATE_LINK ${CMAKE_SOURCE_DIR}/src/stdlib ${STDLIB_TMP_PATH}/stdlib
     COPY_ON_ERROR SYMBOLIC)

target_link_libraries(verona-mlir PRIVATE ${LIBS})
install(TARGETS verona-mlir RUNTIME DESTINATION .)

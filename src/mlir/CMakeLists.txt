find_package(Threads REQUIRED)

# We're not building LLVM, but downloading the build dir from an artefact
set(LLVM_BUILD ${VERONA_LLVM_LOCATION})

# std::iterator is used by MLIR/LLVM
# This is deprecated and causes warnings in MSVC.
# Once fixed in LLVM, the following three lines can be removed.
if (MSVC)
  add_compile_definitions(_SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING)
endif ()

# LLVM requires CMake 3.13+ and this policy is for 3.12-, so we force new to
# avoid the CMake warnings on LLVM CMake files that we can't change
cmake_policy(SET CMP0077 NEW)

# Find MLIR_DIR and LLVM_EXTERNAL_LIT from LLVM_BUILD
set(LLVM_DIR ${LLVM_BUILD}/lib/cmake/llvm)
message(STATUS "Setting LLVM_DIR as ${LLVM_DIR}")
set(MLIR_DIR ${LLVM_BUILD}/lib/cmake/mlir)
message(STATUS "Setting MLIR_DIR as ${MLIR_DIR}")
set(LLVM_EXTERNAL_LIT ${LLVM_BUILD}/bin/llvm-lit CACHE STRING "Command used to spawn lit")
message(STATUS "Setting LLVM_EXTERNAL_LIT as ${LLVM_EXTERNAL_LIT}")

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

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(../ds)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/mlir)
include_directories(${PROJECT_BINARY_DIR}/mlir)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

#add_subdirectory(dialect)

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
add_llvm_executable(verona-mlir
  consumer.cc
  driver.cc
  error.cc
  generator.cc
  verona-mlir.cc)

llvm_update_compile_flags(verona-mlir)
target_link_libraries(verona-mlir PRIVATE ${LIBS})
install(TARGETS verona-mlir RUNTIME DESTINATION .)

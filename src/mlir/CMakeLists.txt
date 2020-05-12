find_package(Threads REQUIRED)

# We're not building LLVM, but downloading the build dir from an artefact
set(LLVM_BUILD "${CMAKE_SOURCE_DIR}/external/llvm-project/build")

# Find MLIR_DIR and LLVM_EXTERNAL_LIT from LLVM_BUILD
if (NOT DEFINED MLIR_DIR AND DEFINED LLVM_BUILD)
  set(LLVM_DIR ${LLVM_BUILD}/lib/cmake/llvm)
  message(STATUS "Setting LLVM_DIR as ${LLVM_DIR}")
  set(MLIR_DIR ${LLVM_BUILD}/lib/cmake/mlir)
  message(STATUS "Setting MLIR_DIR as ${MLIR_DIR}")
endif()
if (NOT DEFINED LLVM_EXTERNAL_LIT AND DEFINED LLVM_BUILD)
  set(LLVM_EXTERNAL_LIT ${LLVM_BUILD}/bin/llvm-lit CACHE STRING "Command used to spawn lit")
  message(STATUS "Setting LLVM_EXTERNAL_LIT as ${LLVM_EXTERNAL_LIT}")
endif()
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Enable exception handling
set(LLVM_REQUIRES_EH ON)
set(LLVM_REQUIRES_RTTI ON)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src/mlir)
include_directories(${PROJECT_BINARY_DIR}/src/mlir)
include_directories(${PROJECT_SOURCE_DIR}/external/CLI11/include)
include_directories(${PROJECT_SOURCE_DIR}/external/cpp-peglib)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(dialect)

### Verona Opt (MLIR-to-MLIR/LLVM pipeline)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRVerona
        MLIRTargetLLVMIR
        verona-ast-lib
        )
add_llvm_executable(verona-mlir verona-mlir.cc generator.cc ast-utils.cc)

llvm_update_compile_flags(verona-mlir)
target_link_libraries(verona-mlir PRIVATE ${LIBS} CLI11::CLI11)
install(TARGETS verona-mlir RUNTIME DESTINATION .)

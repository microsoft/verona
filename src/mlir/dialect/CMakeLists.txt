set(LLVM_TARGET_DEFINITIONS VeronaOps.td)
mlir_tablegen(VeronaOps.h.inc -gen-op-decls)
mlir_tablegen(VeronaOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS VeronaTypes.td)
mlir_tablegen(VeronaTypes.h.inc -gen-typedef-decls)
mlir_tablegen(VeronaTypes.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS VeronaDialect.td)
mlir_tablegen(VeronaDialect.h.inc -gen-dialect-decls -dialect=verona)

add_public_tablegen_target(VeronaIncGen)

add_mlir_interface(TypecheckInterface)

add_mlir_dialect_library(MLIRVerona
    VeronaDialect.cc
    VeronaOps.cc
    VeronaTypes.cc
    Typechecker.cc
    TypecheckInterface.cc
    TypeSyntax.cc

    DEPENDS
    VeronaIncGen
    MLIRTypecheckInterfaceIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    )

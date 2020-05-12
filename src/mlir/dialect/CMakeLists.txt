add_mlir_dialect(VeronaOps verona)
add_mlir_doc(VeronaDialect -gen-dialect-doc VeronaDialect Verona/)
add_mlir_doc(VeronaOps -gen-op-doc VeronaOps Verona/)

add_mlir_dialect_library(MLIRVerona
    VeronaDialect.cc
    VeronaOps.cc

    DEPENDS
    MLIRVeronaOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    )

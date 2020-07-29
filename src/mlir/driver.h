#pragma once

#include "ast/ast.h"
#include "error.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::verona
{
  /**
   * Main compiler API.
   *
   * The driver is used by calling three methods in succession:
   * - `readXXX` is used to load the MLIR module. It can either be loaded from
   *    textual MLIR or from the AST.
   * - `process` runs the MLIR lowering pipeline. Passes are selected by the
   *    Driver's constructor's arguments.
   * - `emitXXX` writes the result to a text file, either in MLIR or in LLVM IR
   *    format.
   *
   * For now, the error handling is crude and needs proper consideration,
   * especially aggregating all errors and context before sending it back to
   * the public API callers.
   */
  class Driver
  {
  public:
    Driver(unsigned optLevel = 0, bool lowerToLLVM = false);

    // TODO: add a readSource function that parses Verona source code.
    // this might be more thinking about the error API of the Driver.

    /// Lower an AST into an MLIR module, which is loaded in the driver.
    llvm::Error readAST(const ::ast::Ast& ast);

    /// Read textual MLIR into the driver's module.
    llvm::Error readMLIR(const std::string& filename);

    // Run the MLIR pipeline on the module. This uses the passes that were
    // configured by the driver's constructor.
    llvm::Error process();

    /// Emit the module as textual LLVM IR.
    /// This will fail if the module is not in LLVM dialect yet.
    llvm::Error emitLLVM(const llvm::StringRef filename);

    /// Emit the module as textual MLIR.
    llvm::Error emitMLIR(const llvm::StringRef filename);

  private:
    /// MLIR context.
    mlir::MLIRContext context;

    /// MLIR module.
    /// It gets modified as the driver progresses through its passes.
    mlir::OwningModuleRef module;

    /// MLIR Pass Manager
    /// It gets configured by the constructor based on the provided arguments.
    mlir::PassManager passManager;
  };
}

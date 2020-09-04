// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "ast/ast.h"
#include "dialect/VeronaDialect.h"
#include "error.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/SourceMgr.h"

namespace mlir::verona
{
  /**
   * Main compiler API.
   *
   * The driver is user by first calling one of the `readXXX` methods, followed
   * by `emitMLIR`. The various `readXXX` methods allow using different kinds of
   * input.
   *
   * The lowering pipeline is configured through Driver's constructor arguments.
   *
   * For now, the error handling is crude and needs proper consideration,
   * especially aggregating all errors and context before sending it back to
   * the public API callers.
   */
  class Driver
  {
  public:
    Driver(unsigned optLevel = 0);

    // TODO: add a readSource function that parses Verona source code.
    // this might be more thinking about the error API of the Driver.

    /// Lower an AST into an MLIR module, which is loaded in the driver.
    llvm::Error readAST(const ::ast::Ast& ast);

    /// Read textual MLIR into the driver's module.
    llvm::Error readMLIR(const std::string& filename);

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

    /// Source manager.
    llvm::SourceMgr sourceManager;

    /// Diagnostic handler that pretty-prints MLIR errors.
    /// The handler registers itself with the MLIR context and gets invoked
    /// automatically. We only need to keep it alive by storing it here.
    SourceMgrDiagnosticHandler diagnosticHandler;
  };
}

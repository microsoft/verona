// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "error.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "parser/ast.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/SourceMgr.h"

namespace mlir::verona
{
  /**
   * Main compiler API.
   *
   * The two entry points are `readAST` and `readMLIR`.
   * The intermediate steps are `optimiseMLIR` and `loweToLLVM`.
   * The final step is `codeGeneration`.
   * The testing methods are `emitMLIR`, `emitLLVM` and `runLLVM`.
   *
   * In ASCii-art:
   *
   *   readAST -->.<-- readMLIR(file)
   *              |
   *              +--> emitMLIR(file)
   *              |
   *             ---
   *         optimiseMLIR
   *             ---
   *              |
   *              +--> emitMLIR(file)
   *              |
   *             ---
   *          lowerToLLVM
   *             ---
   *              |
   *              +--> emitLLVM(file)
   *              +--> runLLVM()
   *              |
   *              `--> codeGeneration(file)
   *
   * For now, the error handling is crude and needs proper consideration,
   * especially aggregating all errors and context before sending it back to
   * the public API callers.
   */
  class Driver
  {
  public:
    Driver(unsigned optLevel = 0);

    /// Lower an AST into an MLIR module, which is loaded in the driver.
    /// Populates mlirModule
    llvm::Error readAST(::verona::parser::Ast ast);

    /// Read textual MLIR into the driver's module.
    /// Populates mlirModule
    llvm::Error readMLIR(const std::string& filename);

    /// Emit the module as textual MLIR.
    /// Requires mlirModule
    /// For testing purposes only
    llvm::Error emitMLIR(const llvm::StringRef filename);

    /// Optimise the MLIR module in preparation for LLVM lowering
    /// Requires mlirModule
    llvm::Error optimiseMLIR();

    /// Lower to LLVM IR (+ basic opt passes)
    /// Requires mlirModule
    /// Populates llvmModule
    llvm::Error lowerToLLVM();

    /// Emit the module as textual LLVM IR.
    /// Requires llvmModule
    /// For testing purposes only
    llvm::Error emitLLVM(const llvm::StringRef filename);

    /// JIT & execute the module and print the return value of main.
    /// Requires llvmModule
    /// For testing purposes only
    llvm::Error runLLVM(int& returnValue);

    /// Emits the object code for the target
    /// TODO: Implement this functionality, needs target-triple, etc.
    /// Argument is the final object filename
    llvm::Error codeGeneration(const llvm::StringRef filename);

  private:
    /// MLIR context.
    mlir::MLIRContext mlirContext;

    /// LLVMContext.
    std::unique_ptr<llvm::LLVMContext> llvmContext;

    /// MLIR module.
    /// It gets modified as the driver progresses through its passes.
    mlir::OwningModuleRef mlirModule;

    /// LLVM module.
    /// It gets modified as the driver progresses through its passes.
    std::unique_ptr<llvm::Module> llvmModule;

    /// Optimisation level (for both MLIR and LLVM IRs)
    unsigned optLevel;
    bool mlirOptimised = false;

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

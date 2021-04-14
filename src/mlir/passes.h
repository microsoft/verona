// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir
{
  /**
   * Verona Mangling Pass
   *
   * This pass mangles the name of all functions inside modules (classes,
   * modules) and move them into the top module, replacing their uses with the
   * new functions (on calls). This allows us to lower the MLIR module into LLVM
   * dialect and then LLVM IR.
   *
   * The original module has nested modules to make it easier to build IR with
   * equal names for different implementaitons (for example, a type's `apply`
   * function). Once mangled and moved to the top module, all those methods will
   * have a unique name, identified by their classes and parents.
   *
   * This pass is mandatory to lower to LLVM IR.
   */
  class VeronaManglingPass : public OperationPass<ModuleOp>
  {
    /// Top module, to move the mangled symbols to
    ModuleOp topModule;

    /// Current sub-module, mangled as __mod1__mod2__...__modn__(symbol)
    llvm::SmallVector<llvm::StringRef, 4> moduleStack;

    /// List of modules to delete after to avoid invalidating iterators
    llvm::SmallVector<ModuleOp, 4> modulesToRemove;

    /// Mangle the name using the module stack
    std::string mangleName(llvm::StringRef funcName);

    /// Recursive function that scans symbols, mangles names, moves to global
    /// module.
    void runOnModule(ModuleOp mod);

  public:
    VeronaManglingPass()
    : OperationPass<ModuleOp>(TypeID::get<VeronaManglingPass>())
    {}

    /// Runs the mangling pass on the module
    void runOnOperation() override;
    /// Pass name
    llvm::StringRef getName() const override
    {
      return "VeronaManglingPass";
    }
    /// Clone pass
    std::unique_ptr<Pass> clonePass() const override
    {
      return std::make_unique<VeronaManglingPass>();
    }
  };

  /// Creates a pass to mangle all function and classes names in order to
  /// flatten the module structure so that the LLVM lower can work
  std::unique_ptr<OperationPass<ModuleOp>> createVeronaManglingPass();
}

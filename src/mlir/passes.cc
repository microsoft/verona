// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "passes.h"

namespace mlir
{
  std::string VeronaManglingPass::mangleName(llvm::StringRef funcName)
  {
    // TODO: This is inefficient but works for now
    std::string name;
    for (auto modName : moduleStack)
    {
      name += modName.str() + "__";
    }
    name += funcName;
    return name;
  }

  void VeronaManglingPass::runOnModule(ModuleOp mod)
  {
    // Push the module name on the stack
    moduleStack.push_back(mod.getName().getValue());
    auto depth = moduleStack.size();

    // Each module can only have one region, but the region can have multiple
    // sub-modules.
    auto& region = mod->getRegion(0);

    // Only move functions down when in a submodule
    if (depth > 1)
    {
      // For all functions
      for (auto func : region.getOps<FuncOp>())
      {
        // Calls aren't replaced with replaceAllUsersWith, certainly not across
        // regions, so we need a better strategy. For now, we only handle
        // functions with unique names and ignore the scope, but we'll need a
        // better strategy soon.
        auto newFunc = func.clone();
        // auto mangledName = mangleName(func.getName());
        // newFunc.setName(mangledName);
        topModule.push_back(newFunc);
        // func->replaceAllUsesWith(newFunc);
      }
    }

    // Recurse for all sub-modules
    for (auto submod : region.getOps<ModuleOp>())
    {
      runOnModule(submod);
    }

    // Module now should be empty and can be deleted
    if (depth > 1)
    {
      modulesToRemove.push_back(mod);
    }

    // Pop the name off the stack
    moduleStack.resize(moduleStack.size() - 1);
  }

  void VeronaManglingPass::runOnOperation()
  {
    topModule = getOperation();
    runOnModule(topModule);

    for (auto mod : modulesToRemove)
    {
      mod->erase();
    }
  }

  std::unique_ptr<OperationPass<ModuleOp>> createVeronaManglingPass()
  {
    return std::make_unique<VeronaManglingPass>();
  }
}

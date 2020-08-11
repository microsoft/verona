// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "driver.h"

#include "dialect/Typechecker.h"
#include "generator.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir::verona
{
  Driver::Driver(unsigned optLevel)
  : passManager(&context), diagnosticHandler(sourceManager, &context)
  {
    // Opaque operations and types can only exist if we allow
    // unregistered dialects to co-exist. Full conversions later will
    // make sure we end up with only Verona dialect, then Standard
    // dialects, then LLVM dialect, before converting to LLVM IR.
    context.allowUnregisteredDialects();

    // TODO: make the set of passes configurable from the command-line
    passManager.addPass(std::make_unique<TypecheckerPass>());

    if (optLevel > 0)
    {
      passManager.addPass(mlir::createInlinerPass());
      passManager.addPass(mlir::createSymbolDCEPass());

      mlir::OpPassManager& funcPM = passManager.nest<mlir::FuncOp>();
      funcPM.addPass(mlir::createCanonicalizerPass());
      funcPM.addPass(mlir::createCSEPass());
    }
  }

  llvm::Error Driver::readAST(const ::ast::Ast& ast)
  {
    auto result = Generator::lower(&context, ast);
    if (!result)
      return result.takeError();

    module = std::move(*result);

    if (failed(verify(*module)))
    {
      module->dump();
      return runtimeError("AST was lowered to invalid MLIR");
    }

    return llvm::Error::success();
  }

  llvm::Error Driver::readMLIR(const std::string& filename)
  {
    if (filename.empty())
      return runtimeError("No input filename provided");

    // Read an MLIR file
    auto srcOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filename);

    if (auto err = srcOrErr.getError())
      return runtimeError(
        "Cannot open file " + filename + ": " + err.message());

    // Add the input to the source manager and parse it.
    // `parseSourceFile` already includes verification of the IR.
    sourceManager.AddNewSourceBuffer(std::move(*srcOrErr), llvm::SMLoc());
    module = mlir::parseSourceFile(sourceManager, &context);
    if (!module)
      return runtimeError("Can't load MLIR file");

    return llvm::Error::success();
  }

  llvm::Error Driver::emitMLIR(llvm::StringRef filename)
  {
    assert(module);

    if (filename.empty())
      return runtimeError("No output filename provided");

    if (failed(passManager.run(module.get())))
    {
      module->dump();
      return runtimeError("Failed to run some passes");
    }

    // Write to the file requested
    std::error_code error;
    auto out = llvm::raw_fd_ostream(filename, error);
    if (error)
      return runtimeError("Cannot open output filename");

    module->print(out);
    return llvm::Error::success();
  }
}

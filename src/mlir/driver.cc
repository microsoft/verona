// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "driver.h"

#include "consumer.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir::verona
{
  Driver::Driver(unsigned optLevel)
  : optLevel(optLevel),
    passManager(&context),
    diagnosticHandler(sourceManager, &context)
  {
    // These are the dialects we emit directly
    context.getOrLoadDialect<mlir::StandardOpsDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

    // Some simple MLIR optimisations
    if (optLevel > 0)
    {
      passManager.addPass(mlir::createInlinerPass());
      passManager.addPass(mlir::createSymbolDCEPass());

      mlir::OpPassManager& funcPM = passManager.nest<mlir::FuncOp>();
      funcPM.addPass(mlir::createCanonicalizerPass());
      funcPM.addPass(mlir::createCSEPass());
    }
  }

  llvm::Error Driver::readAST(::verona::parser::Ast ast)
  {
    // Use the MLIR generator to lower the AST into MLIR
    auto result = ASTConsumer::lower(&context, ast);
    if (!result)
      return result.takeError();
    module = std::move(*result);

    // Verify the module to make sure we didn't do anything silly
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

    // Read an MLIR file into a buffer
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

    // If optimisation levels higher than 0, run some opts
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

  llvm::Error Driver::emitLLVM(llvm::StringRef filename)
  {
    if (filename.empty())
      return runtimeError("No output filename provided");

    // The lowering "pass manager"
    passManager.addPass(mlir::createLowerToLLVMPass());

    // If optimisation levels higher than 0, run some opts
    if (mlir::failed(passManager.run(module.get())))
    {
      module->dump();
      return runtimeError("Failed to run some passes");
    }

    // Register the translation to LLVM IR with the MLIR context.
    mlir::registerLLVMDialectTranslation(*module->getContext());

    // Then lower to LLVM IR (via LLVM dialect)
    llvm::LLVMContext llvmContext;
    auto llvm = mlir::translateModuleToLLVMIR(module.get(), llvmContext);
    if (!llvm)
      return runtimeError("Failed to lower to LLVM IR");

    // Optimise if requested
    if (optLevel)
    {
      // Initialize LLVM targets.
      // TODO: Use target triples here, for cross-compilation
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter();
      mlir::ExecutionEngine::setupTargetTriple(llvm.get());

      /// Optionally run an optimization pipeline over the llvm module.
      auto optPipeline = mlir::makeOptimizingTransformer(
        optLevel,
        /*sizeLevel=*/0,
        /*targetMachine=*/nullptr);
      if (auto err = optPipeline(llvm.get()))
        return runtimeError("Failed to generate LLVM IR");
    }

    // Write to the file requested
    std::error_code error;
    auto out = llvm::raw_fd_ostream(filename, error);
    if (error)
      return runtimeError("Failed open output file");
    llvm->print(out, nullptr);

    return llvm::Error::success();
  }
}

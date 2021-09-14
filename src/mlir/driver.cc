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
#include "orcjit.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

#include <iostream>

namespace
{
  /// Return the correct output stream for writing operations.
  /// Possible returns are: file, stdout or null
  llvm::Expected<std::unique_ptr<llvm::raw_ostream>>
  getOutputStream(llvm::StringRef filename)
  {
    std::unique_ptr<llvm::raw_ostream> out;
    if (filename.empty())
    {
      // Empty filename is null output
      out = std::make_unique<llvm::raw_null_ostream>();
    }
    else
    {
      // Otherwise, both filename and - do the right thing
      std::error_code error;
      out = std::make_unique<llvm::raw_fd_ostream>(filename, error);
      if (error)
        return mlir::verona::runtimeError("Cannot open output filename");
    }

    return out;
  }
}

namespace mlir::verona
{
  Driver::Driver(unsigned optLevel)
  : optLevel(optLevel),
    passManager(&mlirContext),
    diagnosticHandler(sourceManager, &mlirContext)
  {
    // These are the dialects we emit directly
    mlirContext.getOrLoadDialect<mlir::StandardOpsDialect>();
    mlirContext.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

    // Initialize LLVM targets.
    // TODO: Use target triples here, for cross-compilation
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  }

  llvm::Error Driver::readAST(::verona::parser::Ast ast)
  {
    // Use the MLIR generator to lower the AST into MLIR
    auto result = ASTConsumer::lower(&mlirContext, ast);
    if (!result)
      return result.takeError();
    mlirModule = std::move(*result);

    // Verify the mlirModule to make sure we didn't do anything silly
    if (failed(verify(*mlirModule)))
    {
      mlirModule->dump();
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
    mlirModule = mlir::parseSourceFile(sourceManager, &mlirContext);
    if (!mlirModule)
      return runtimeError("Can't load MLIR file");

    return llvm::Error::success();
  }

  llvm::Error Driver::emitMLIR(llvm::StringRef filename)
  {
    assert(mlirModule);

    auto out = getOutputStream(filename);
    if (auto err = out.takeError())
      return err;

    // Write to the file requested
    mlirModule->print(*out.get());

    return llvm::Error::success();
  }

  llvm::Error Driver::optimiseMLIR()
  {
    assert(mlirModule);

    if (mlirOptimised)
      return llvm::Error::success();

    // Some simple MLIR optimisations
    if (optLevel > 0)
    {
      passManager.addPass(mlir::createInlinerPass());
      passManager.addPass(mlir::createSymbolDCEPass());

      mlir::OpPassManager& funcPM = passManager.nest<mlir::FuncOp>();
      funcPM.addPass(mlir::createCanonicalizerPass());
      funcPM.addPass(mlir::createCSEPass());
    }

    // If optimisation levels higher than 0, run some opts
    if (failed(passManager.run(mlirModule.get())))
    {
      mlirModule->dump();
      return runtimeError("Failed to run some passes");
    }
    mlirOptimised = true;

    return llvm::Error::success();
  }

  llvm::Error Driver::lowerToLLVM()
  {
    assert(mlirModule);

    // The lowering "pass manager"
    passManager.addPass(mlir::createLowerToLLVMPass());

    // If optimisation levels higher than 0, run some opts
    if (mlir::failed(passManager.run(mlirModule.get())))
    {
      mlirModule->dump();
      return runtimeError("Failed to run some passes");
    }

    // Register the translation to LLVM IR with the MLIR mlirContext.
    mlir::registerLLVMDialectTranslation(*mlirModule->getContext());

    // Then lower to LLVM IR (via LLVM dialect)
    if (!llvmContext)
      llvmContext = std::make_unique<llvm::LLVMContext>();
    llvmModule = mlir::translateModuleToLLVMIR(mlirModule.get(), *llvmContext);
    if (!llvmModule)
      return runtimeError("Failed to lower to LLVM IR");

    // Optimise if requested
    if (optLevel)
    {
      mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

      // Optionally run an optimization pipeline over the llvm mlirModule.
      auto optPipeline = mlir::makeOptimizingTransformer(
        optLevel,
        /*sizeLevel=*/0,
        /*targetMachine=*/nullptr);
      if (auto err = optPipeline(llvmModule.get()))
        return runtimeError("Failed to generate LLVM IR");
    }

    return llvm::Error::success();
  }

  llvm::Error Driver::emitLLVM(llvm::StringRef filename)
  {
    if (!llvmModule)
    {
      auto err = lowerToLLVM();
      if (err)
        return err;
    }

    auto out = getOutputStream(filename);
    if (auto err = out.takeError())
      return err;

    // Write to the file requested
    llvmModule->print(*out.get(), nullptr);

    return llvm::Error::success();
  }

  llvm::Error Driver::runLLVM(int& returnValue)
  {
    if (!llvmModule)
    {
      auto err = lowerToLLVM();
      if (err)
        return err;
    }
    llvm::ExitOnError check;

    auto J = llvm::orc::VeronaJIT::Create();
    if (!J)
      return J.takeError();
    auto& JIT = *J;
    llvm::orc::ThreadSafeModule TSM(
      std::move(llvmModule), std::move(llvmContext));
    check(JIT->addModule(std::move(TSM)));
    auto MainSymbol = JIT->lookup("main");
    if (!MainSymbol)
      return MainSymbol.takeError();
    auto* Main = (int (*)(int, char*[]))MainSymbol->getAddress();
    returnValue = Main(0, {});

    return llvm::Error::success();
  }

  llvm::Error Driver::codeGeneration(const llvm::StringRef filename)
  {
    return runtimeError("Not implemented yet");
  }
}

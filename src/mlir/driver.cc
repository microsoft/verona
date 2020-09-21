// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "driver.h"

#include "dialect/Typechecker.h"
#include "generator.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir::verona
{
  Driver::Driver(unsigned optLevel, bool enableDiagnosticsVerifier)
  : context(/*loadAllDialects=*/false),
    passManager(&context),
    enableDiagnosticsVerifier(enableDiagnosticsVerifier)
  {
    if (enableDiagnosticsVerifier)
      diagnosticHandler = std::make_unique<SourceMgrDiagnosticVerifierHandler>(
        sourceManager, &context);
    else
      diagnosticHandler =
        std::make_unique<SourceMgrDiagnosticHandler>(sourceManager, &context);

    // In diagnostics verification mode, don't print the associated operation.
    // It would just be adding noise to the test files
    context.printOpOnDiagnostic(!enableDiagnosticsVerifier);

    context.getOrLoadDialect<mlir::StandardOpsDialect>();
    context.getOrLoadDialect<mlir::verona::VeronaDialect>();

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
      return runtimeError("AST was lowered to invalid MLIR");

    return llvm::Error::success();
  }

  llvm::Error Driver::readMLIR(std::unique_ptr<llvm::MemoryBuffer> buffer)
  {
    // Add the input to the source manager and parse it.
    // `parseSourceFile` already includes verification of the IR.
    sourceManager.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
    module = mlir::parseSourceFile(sourceManager, &context);
    if (!module)
      return runtimeError("Can't load MLIR file");

    return llvm::Error::success();
  }

  llvm::Error Driver::emitMLIR(llvm::raw_ostream& os)
  {
    assert(module);

    if (failed(passManager.run(module.get())))
      return runtimeError("Failed to run some passes");

    module->print(os);
    return llvm::Error::success();
  }

  llvm::Error Driver::verifyDiagnostics()
  {
    assert(enableDiagnosticsVerifier);

    auto* handler =
      static_cast<SourceMgrDiagnosticVerifierHandler*>(diagnosticHandler.get());

    if (failed(handler->verify()))
      return runtimeError("Diagnostic verification failed\n");

    return llvm::Error::success();
  }

  void Driver::dumpMLIR(llvm::raw_ostream& os)
  {
    if (module)
      module->print(os);
  }
}

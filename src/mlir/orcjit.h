// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"

#include <memory>

namespace llvm::orc
{
  /**
   * Verona JIT Executor
   *
   * Copied from KaleidoscopeJIT.h, has a very dumb execution model.
   *
   * The main purpose of this helper is for testing, as the actual execution
   * model will be full object code generation and native execution.
   *
   * 1. Create the JIT via static function:
   *    auto J = VeronaJIT::Create();
   *
   * 2. Provide a thread-safe LLVM module:
   *    ThreadSafeModule TSM(mod, context);
   *    J->addModule(TSM);
   *
   * 3. Lookup the "main" symbol (can be any valid symbol):
   *    auto M = J->lookup("main");
   *
   * 4. Cast to a function pointer:
   *    auto* f = (int (*)(int, char*[]))M->getAddress()
   *
   * 5. Call, capturing the return value:
   *    int ret = f(argc, argv);
   */
  class VeronaJIT
  {
  private:
    std::unique_ptr<ExecutionSession> ES;

    DataLayout DL;
    MangleAndInterner Mangle;

    RTDyldObjectLinkingLayer ObjectLayer;
    IRCompileLayer CompileLayer;

    JITDylib& MainJD;

  public:
    VeronaJIT(
      std::unique_ptr<ExecutionSession> ES,
      JITTargetMachineBuilder JTMB,
      DataLayout DL)
    : ES(std::move(ES)),
      DL(std::move(DL)),
      Mangle(*this->ES, this->DL),
      ObjectLayer(
        *this->ES, []() { return std::make_unique<SectionMemoryManager>(); }),
      CompileLayer(
        *this->ES,
        ObjectLayer,
        std::make_unique<ConcurrentIRCompiler>(std::move(JTMB))),
      MainJD(this->ES->createBareJITDylib("<main>"))
    {
      MainJD.addGenerator(
        cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(
          DL.getGlobalPrefix())));
      // Needed for COFF targets
      ObjectLayer.setOverrideObjectFlagsWithResponsibilityFlags(true);
    }

    ~VeronaJIT()
    {
      if (auto Err = ES->endSession())
        ES->reportError(std::move(Err));
    }

    static Expected<std::unique_ptr<VeronaJIT>> Create()
    {
      auto EPC = SelfExecutorProcessControl::Create();
      if (!EPC)
        return EPC.takeError();

      auto ES = std::make_unique<ExecutionSession>(std::move(*EPC));

      JITTargetMachineBuilder JTMB(
        ES->getExecutorProcessControl().getTargetTriple());

      auto DL = JTMB.getDefaultDataLayoutForTarget();
      if (!DL)
        return DL.takeError();

      return std::make_unique<VeronaJIT>(
        std::move(ES), std::move(JTMB), std::move(*DL));
    }

    const DataLayout& getDataLayout() const
    {
      return DL;
    }

    JITDylib& getMainJITDylib()
    {
      return MainJD;
    }

    Error addModule(ThreadSafeModule TSM, ResourceTrackerSP RT = nullptr)
    {
      if (!RT)
        RT = MainJD.getDefaultResourceTracker();

      // Make sure the module has a triple+DL
      // If not, assume the same as the JIT
      auto mod = TSM.getModuleUnlocked();
      if (mod->getTargetTriple().empty())
        mod->setTargetTriple(
          ES->getExecutorProcessControl().getTargetTriple().getTriple());
      if (mod->getDataLayout().isDefault())
        mod->setDataLayout(DL.getStringRepresentation());
      return CompileLayer.add(RT, std::move(TSM));
    }

    Expected<JITEvaluatedSymbol> lookup(StringRef Name)
    {
      return ES->lookup({&MainJD}, Mangle(Name.str()));
    }
  };
} // end namespace llvm::orc

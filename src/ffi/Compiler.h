// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "CXXType.h"
#include "FS.h"

#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/CodeGen/ModuleBuilder.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Frontend/MultiplexConsumer.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Frontend/Utils.h>
#include <clang/Lex/HeaderSearch.h>
#include <clang/Lex/HeaderSearchOptions.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <clang/Sema/Sema.h>
#include <clang/Sema/Template.h>
#include <clang/Sema/TemplateDeduction.h>
#include <clang/Serialization/ASTWriter.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>

using namespace clang;

namespace verona::ffi
{
  /// Source languages Clang supports
  enum SourceLanguage
  {
    C,
    CXX,
    ObjC,
    ObjCXX,
    SOURCE_LANGAUGE_ENUM_SIZE
  };

  /**
   * Compiler wrapper. Boilerplate for Clang specific logic to simplify
   * CXXInterface to only AST logic.
   */
  class Compiler
  {
    /// Pre-compiled header
    llvm::Optional<clang::PrecompiledPreamble> preamble;
    /// LLVMContext for LLVM lowering.
    std::unique_ptr<llvm::LLVMContext> llvmContext{new llvm::LLVMContext};
    /// Compiler instance.
    std::unique_ptr<CompilerInstance> Clang;
    /// All arguments, including empty last one that is replaced evey call
    std::vector<const char*> args;

    /// Converts SourceLanguage into string
    static const char* source_language_string(SourceLanguage sl)
    {
      static std::array<const char*, SOURCE_LANGAUGE_ENUM_SIZE> names = {
        "c", "c++", "objective-c", "objective-c++"};
      return names.at(static_cast<int>(sl));
    }

    /// Returns the array with the filename as the last arg
    llvm::ArrayRef<const char*> getArgs(const char* filename)
    {
      args[args.size() - 1] = filename;
      return args;
    }

  public:
    /**
     * Creates the Clang instance, with preprocessor and header search support.
     *
     * FIXME: Do something more sensible with the diagnostics engine so
     * that we can propagate errors to Verona.
     */
    Compiler(
      llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
      const char* sourceName,
      SourceLanguage sourceLang = SourceLanguage::CXX)
    {
      // Initilise the default arguments plus space for the filename
      const char* langName = source_language_string(sourceLang);
      // FIXME: Don't hard code include paths!
      args = {
        "clang",
        "-x",
        langName,
        "-I",
        "/usr/include/",
        "-I",
        "/usr/local/include/",
        ""};

      // Compiler Instance
      Clang = std::make_unique<CompilerInstance>();

      // Diagnostics
      // TODO: Wire up diagnostics so that we can spot invalid template
      // instantiations.
      IntrusiveRefCntPtr<DiagnosticIDs> DiagID = new DiagnosticIDs();
      IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
      auto* DiagsPrinter = new TextDiagnosticPrinter{llvm::errs(), &*DiagOpts};
      auto* Diags =
        new DiagnosticsEngine(DiagID, DiagOpts, DiagsPrinter, false);

      // Compiler invocation
      auto CI = createInvocationFromCommandLine(
        getArgs(sourceName), Diags, llvm::vfs::getRealFileSystem());

      // File and source manager
      // FIXME: A new Diags is needed here because the invocation takes
      // ownership. Can we make this more obvious?
      Diags = new DiagnosticsEngine(DiagID, DiagOpts, DiagsPrinter, false);

      // Create the file manager
      // NOTE: Both managers pointers will be owned by CompilerInstance
      auto* fileManager =
        new clang::FileManager(clang::FileSystemOptions{}, FS);
      auto* sourceManager = new clang::SourceManager(
        *Diags,
        *fileManager,
        /*UserFilesAreVolatile*/ false);
      Clang->setFileManager(fileManager);
      Clang->setSourceManager(sourceManager);
      Clang->setInvocation(std::move(CI));
      Clang->setDiagnostics(Diags);

      // Pre-processor and header search
      auto PPOpts = std::make_shared<PreprocessorOptions>();
      TrivialModuleLoader TML;
      auto HeaderSearchPtr = std::make_unique<HeaderSearch>(
        std::make_shared<HeaderSearchOptions>(),
        Clang->getSourceManager(),
        *Diags,
        Clang->getLangOpts(),
        nullptr);
      auto PreprocessorPtr = std::make_shared<Preprocessor>(
        PPOpts,
        *Diags,
        Clang->getLangOpts(),
        Clang->getSourceManager(),
        *HeaderSearchPtr,
        TML,
        nullptr,
        false);
      Clang->setPreprocessor(PreprocessorPtr);
      Clang->getPreprocessor().enableIncrementalProcessing();
    }

    /**
     * Set the AST machinery to allow consumers to traverse the graph.
     *
     * FIXME: This came from CXXInterface c-tor and should probably be more
     * factored around the corners.
     */
    void setASTMachinery(
      std::unique_ptr<clang::ASTConsumer> consumer, clang::ASTContext* ast)
    {
      Clang->setASTConsumer(std::move(consumer));
      Clang->setASTContext(ast);
      Clang->createSema(TU_Complete, nullptr);
    }

    /**
     * Get a location at the end of file (for new code)
     *
     * FIXME: This came from CXXInterface c-tor and should probably be more
     * factored around the corners.
     */
    SourceLocation getEndOfFileLocation() const
    {
      auto mainFile = Clang->getSourceManager().getMainFileID();
      return Clang->getSourceManager().getLocForEndOfFile(mainFile);
    }

    /**
     * Call a FrontendAction on ClangInterface
     *
     * FIXME: This came from CXXInterface c-tor and should probably be more
     * factored around the corners.
     */
    void ExecuteAction(clang::FrontendAction& action)
    {
      Clang->ExecuteAction(action);
    }

    /**
     * Lowers each top-level declaration to LLVM IR and dumps the module.
     *
     * FIXME: This came from CXXInterface c-tor and should probably be more
     * factored around the corners.
     */
    std::unique_ptr<llvm::Module>
    emitLLVM(clang::ASTContext* ast, const char* cu_name)
    {
      std::unique_ptr<CodeGenerator> CodeGen{CreateLLVMCodeGen(
        Clang->getDiagnostics(),
        cu_name,
        Clang->getHeaderSearchOpts(),
        Clang->getPreprocessorOpts(),
        Clang->getCodeGenOpts(),
        *llvmContext)};
      CodeGen->Initialize(*ast);
      CodeGen->HandleTranslationUnit(*ast);
      for (auto& D : ast->getTranslationUnitDecl()->decls())
        CodeGen->HandleTopLevelDecl(DeclGroupRef{D});
      std::unique_ptr<llvm::Module> M{CodeGen->ReleaseModule()};
      return M;
    }

    // Exposing some functionality to make this work
    // TODO: Fix the layering issues

    /// Get Semantic analysis
    clang::Sema& getSema()
    {
      return Clang->getSema();
    }
  };
} // namespace verona::ffi

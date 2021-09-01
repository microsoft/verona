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

namespace verona::interop
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
   * Compiler wrapper.
   *
   * Boilerplate for Clang specific logic to simplify CXXInterface to only AST
   * logic.
   */
  class Compiler
  {
    /// Pre-compiled header
    llvm::Optional<clang::PrecompiledPreamble> preamble;
    /// LLVMContext for LLVM lowering.
    std::unique_ptr<llvm::LLVMContext> llvmContext{new llvm::LLVMContext};
    /// Compiler instance.
    std::unique_ptr<clang::CompilerInstance> Clang;
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
      llvm::ArrayRef<std::string> includePath,
      SourceLanguage sourceLang = SourceLanguage::CXX)
    {
      // Initilise the default arguments plus space for the filename
      const char* langName = source_language_string(sourceLang);
      // Create base command-line
      args = {"clang",
              "-x",
              langName,
              "-I",
              "/usr/include/",
              "-I",
              "/usr/local/include/"};
      // Add user include paths
      for (auto& dir : includePath)
      {
        args.push_back("-I");
        args.push_back(dir.c_str());
      }
      // Add final space for the compile unit name
      args.push_back("");

      // Compiler Instance
      Clang = std::make_unique<clang::CompilerInstance>();

      // Diagnostics
      clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagID =
        new clang::DiagnosticIDs();
      clang::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOpts =
        new clang::DiagnosticOptions();
      auto DiagsPrinter =
        new clang::TextDiagnosticPrinter(llvm::errs(), DiagOpts.get());
      clang::IntrusiveRefCntPtr<clang::DiagnosticsEngine> Diags =
        new clang::DiagnosticsEngine(DiagID, DiagOpts, DiagsPrinter, true);

      // Compiler invocation
      auto CI = createInvocationFromCommandLine(
        getArgs(sourceName), Diags, llvm::vfs::getRealFileSystem());

      // File and source manager
      Clang->setSourceManager(new clang::SourceManager(
        *Diags,
        *new clang::FileManager(clang::FileSystemOptions{}, FS),
        /*UserFilesAreVolatile*/ false));
      Clang->setFileManager(&Clang->getSourceManager().getFileManager());
      Clang->setDiagnostics(Diags.get());
      Clang->setInvocation(std::move(CI));

      // Pre-processor and header search
      auto PPOpts = std::make_shared<clang::PreprocessorOptions>();
      clang::TrivialModuleLoader TML;
      auto HeaderSearchPtr = std::make_unique<clang::HeaderSearch>(
        std::make_shared<clang::HeaderSearchOptions>(),
        Clang->getSourceManager(),
        *Diags,
        Clang->getLangOpts(),
        nullptr);
      auto PreprocessorPtr = std::make_shared<clang::Preprocessor>(
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
      Clang->createSema(clang::TU_Complete, nullptr);
    }

    /**
     * Get a location at the end of file (for new code)
     *
     * FIXME: This came from CXXInterface c-tor and should probably be more
     * factored around the corners.
     */
    clang::SourceLocation getEndOfFileLocation() const
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
    bool ExecuteAction(clang::FrontendAction& action)
    {
      return Clang->ExecuteAction(action);
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
      // Initialise codegen
      std::unique_ptr<clang::CodeGenerator> CodeGen{CreateLLVMCodeGen(
        Clang->getDiagnostics(),
        cu_name,
        Clang->getHeaderSearchOpts(),
        Clang->getPreprocessorOpts(),
        Clang->getCodeGenOpts(),
        *llvmContext)};
      CodeGen->Initialize(*ast);

      // Parse all definitions, including template specialisations
      for (auto& D : ast->getTranslationUnitDecl()->decls())
        CodeGen->HandleTopLevelDecl(clang::DeclGroupRef{D});
      CodeGen->HandleTranslationUnit(*ast);

      // Release the module
      std::unique_ptr<llvm::Module> M{CodeGen->ReleaseModule()};
      return M;
    }

    // Exposing some functionality to make this work
    // TODO: Fix the layering issues

    /// Get Semantic analysis
    clang::Sema& getSema() const
    {
      return Clang->getSema();
    }
  };
} // namespace verona::interop

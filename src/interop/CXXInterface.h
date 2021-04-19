// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "CXXBuilder.h"
#include "CXXQuery.h"
#include "CXXType.h"
#include "Compiler.h"
#include "FS.h"

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
#include <clang/Serialization/ASTWriter.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>

namespace verona::interop
{
  /**
   * C++ Clang Interface.
   *
   * This is the main class for the Clang driver. It collects the needs for
   * reading a source/header file and parsing its AST into a usable format, so
   * that we can use match handlers to find the elements we need from it.
   *
   * There are two main stages:
   *  1. Initialisation: reads the file, parses and generates the pre-compiled
   *     header info, including all necessary headers and files.
   *  2. Query: using match handlers, searches the AST for specific constructs
   *     such as class types, function names, etc.
   *
   */
  class CXXInterface
  {
    /// Name of the internal compilation unit that includes the filename
    static constexpr const char* cu_name = "verona_interface.cc";
    /// The AST root
    clang::ASTContext* ast = nullptr;
    /// Compiler
    std::unique_ptr<Compiler> Clang;
    /// Virtual file system (compiler unit, pch, headers)
    FileSystem FS;
    /// Query system
    std::unique_ptr<CXXQuery> query;
    /// Build system
    std::unique_ptr<CXXBuilder> builder;

    /**
     * Creates new AST consumers to add the AST back into the interface.
     *
     * Use:
     * ```
     *  CompilerInstance->setASTConsumer(factory.newASTConsumer());
     *  CompilerInstance->setASTContext(ast);
     * ```
     */
    struct ASTConsumerFactory
    {
      /// CXX interface
      CXXInterface* interface;
      /// Actual consumer that will be executed.
      struct Collector : public clang::ASTConsumer
      {
        /// CXX interface
        CXXInterface* interface;
        /// Collector C-tor
        Collector(CXXInterface* i) : interface(i) {}
        /// Reassociates the AST back into the interface.
        void HandleTranslationUnit(clang::ASTContext& Ctx) override
        {
          interface->ast = &Ctx;
        }
      };
      /// Factory C-tor
      ASTConsumerFactory(CXXInterface* i) : interface(i) {}
      /// Returns a new unique Collector consumer.
      std::unique_ptr<clang::ASTConsumer> newASTConsumer()
      {
        return std::make_unique<Collector>(interface);
      }
    } factory;

    /**
     * Pre-compiled header action, to create the PCH consumers for PCH
     * generation.
     */
    struct GenerateMemoryPCHAction : clang::GeneratePCHAction
    {
      /// Actual buffer for the PCH, owned externally
      llvm::SmallVectorImpl<char>& outBuffer;
      /// C-tor
      GenerateMemoryPCHAction(llvm::SmallVectorImpl<char>& outBuffer)
      : outBuffer(outBuffer)
      {}
      /// Adds PCH generator, called by Clang->ExecuteAction
      std::unique_ptr<clang::ASTConsumer>
      CreateASTConsumer(clang::CompilerInstance& CI, llvm::StringRef InFile)
      {
        // Check arguments (CI must exist and be initialised)
        std::string Sysroot;
        if (!ComputeASTConsumerArguments(CI, /*ref*/ Sysroot))
        {
          return nullptr;
        }
        const auto& FrontendOpts = CI.getFrontendOpts();

        // Empty filename as we're not reading from disk
        std::string OutputFile;
        // Connect the output stream
        auto OS = std::make_unique<llvm::raw_svector_ostream>(outBuffer);
        // create a buffer
        auto Buffer = std::make_shared<clang::PCHBuffer>();

        // MultiplexConsumer needs a list of consumers
        std::vector<std::unique_ptr<clang::ASTConsumer>> Consumers;

        // PCH generator
        Consumers.push_back(std::make_unique<clang::PCHGenerator>(
          CI.getPreprocessor(),
          CI.getModuleCache(),
          OutputFile,
          Sysroot,
          Buffer,
          FrontendOpts.ModuleFileExtensions,
          false /* Allow errors */,
          FrontendOpts.IncludeTimestamps,
          +CI.getLangOpts().CacheGeneratedPCH));

        // PCH container
        Consumers.push_back(
          CI.getPCHContainerWriter().CreatePCHContainerGenerator(
            CI, InFile.str(), OutputFile, std::move(OS), Buffer));

        return std::make_unique<clang::MultiplexConsumer>(std::move(Consumers));
      }
    };

    /**
     * Generates the pre-compile header into the memory buffer.
     *
     * This method creates a new local Clang just for the pre-compiled headers
     * and returns a memory buffer with the contents, to be inserted in a
     * "file" inside the virtual file system.
     */
    std::unique_ptr<llvm::MemoryBuffer> generatePCH(
      const char* headerFile,
      llvm::ArrayRef<std::string> includePath,
      SourceLanguage sourceLang)
    {
      Compiler LocalClang(
        llvm::vfs::getRealFileSystem(), headerFile, includePath, sourceLang);
      llvm::SmallVector<char, 0> pchOutBuffer;
      auto action = std::make_unique<GenerateMemoryPCHAction>(pchOutBuffer);
      LocalClang.ExecuteAction(*action);
      return std::unique_ptr<llvm::MemoryBuffer>(
        new llvm::SmallVectorMemoryBuffer(std::move(pchOutBuffer)));
    }

  public:
    /**
     * CXXInterface c-tor. Creates the internal compile unit, include the
     * user file (and all dependencies), generates the pre-compiled headers,
     * creates the compiler instance and re-attaches the AST to the interface.
     */
    CXXInterface(
      std::string headerFile,
      llvm::ArrayRef<std::string> includePath,
      SourceLanguage sourceLang = SourceLanguage::CXX)
    : factory(this)
    {
      // Pre-compiles the file requested by the user
      std::unique_ptr<llvm::MemoryBuffer> pchBuffer =
        generatePCH(headerFile.c_str(), includePath, sourceLang);

      // Creating a fake compile unit to include the target file
      // in an in-memory file system.
      std::string Code = "#include \"" + headerFile + "\"\n";
      Code += "namespace verona { namespace __ffi_internal {\n}}\n";
      auto Buf = llvm::MemoryBuffer::getMemBufferCopy(Code);
      auto PCHBuf = llvm::MemoryBuffer::getMemBufferCopy(Code);
      FS.addFile(cu_name, std::move(Buf));

      // Adding the pre-compiler header file to the file system.
      auto pchDataRef = llvm::MemoryBuffer::getMemBuffer(
        llvm::MemoryBufferRef{*pchBuffer}, false);
      FS.addFile(headerFile + ".gch", std::move(pchDataRef));

      // Create the compiler instance and compiles the files
      Clang =
        std::make_unique<Compiler>(FS.get(), cu_name, includePath, sourceLang);

      // Executing the action consumes the AST.  Reset the compiler instance to
      // refer to the AST that it just parsed and create a Sema instance.
      auto collectAST =
        clang::tooling::newFrontendActionFactory(&factory)->create();
      Clang->ExecuteAction(*collectAST);
      Clang->setASTMachinery(factory.newASTConsumer(), ast);

      // Setup query/build system from AST
      query = std::make_unique<CXXQuery>(ast, Clang.get());
      builder = std::make_unique<CXXBuilder>(ast, Clang.get(), query.get());
    }

    /**
     * Dump AST for debug purposes
     */
    void dumpAST() const
    {
      auto* DC = ast->getTranslationUnitDecl();
      DC->dump();
    }

    /**
     * Emit the LLVM code on all generated files
     */
    std::unique_ptr<llvm::Module> emitLLVM()
    {
      return Clang->emitLLVM(ast, cu_name);
    }

    /**
     * Get query system
     */
    const CXXQuery* getQuery() const
    {
      return query.get();
    }

    /**
     * Get builder system
     */
    const CXXBuilder* getBuilder() const
    {
      return builder.get();
    }
  };
} // namespace verona::interop

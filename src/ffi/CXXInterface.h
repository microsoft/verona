// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "CXXType.h"
#include "Compiler.h"
#include "FS.h"
#include "Timer.h"

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
using namespace clang::ast_matchers;

namespace
{
  /// Pre-compiled header action, to create the PCH consumers for PCH generation
  struct GenerateMemoryPCHAction : GeneratePCHAction
  {
    /// Actual buffer for the PCH, owned externally
    llvm::SmallVectorImpl<char>& outBuffer;
    /// C-tor
    GenerateMemoryPCHAction(llvm::SmallVectorImpl<char>& outBuffer)
    : outBuffer(outBuffer)
    {}
    /// Adds PCH generator, called by Clang->ExecuteAction
    std::unique_ptr<ASTConsumer>
    CreateASTConsumer(CompilerInstance& CI, StringRef InFile)
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
      auto Buffer = std::make_shared<PCHBuffer>();

      // MultiplexConsumer needs a list of consumers
      std::vector<std::unique_ptr<ASTConsumer>> Consumers;

      // PCH generator
      Consumers.push_back(std::make_unique<PCHGenerator>(
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

      return std::make_unique<MultiplexConsumer>(std::move(Consumers));
    }
  };
} // namespace anonymous

namespace verona::ffi
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

    /**
     * Creates new AST consumers to add the AST back into the interface.
     *
     * Each traversal consumes the AST, so we need this to add them back
     * for the next operation on the same AST. This is the way to expose
     * the interface pointer so that we can update it again when needed.
     *
     * Use:
     *  CompilerInstance->setASTConsumer(factory.newASTConsumer());
     *  CompilerInstance->setASTContext(ast);
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
        void HandleTranslationUnit(ASTContext& Ctx) override
        {
          fprintf(stderr, "AST consumer %p received AST %p\n", this, &Ctx);
          interface->ast = &Ctx;
        }
        ~Collector()
        {
          fprintf(stderr, "AST consumer %p destroyed\n", this);
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
     * Simple handler for indirect dispatch on a Clang AST matcher.
     *
     * Use:
     *  void myfunc(MatchFinder::MatchResult &);
     *  MatchFinder f;
     *  f.addMatcher(new HandleMatch(myfunc));
     *  f.matchAST(*ast);
     *  // If matches, runs `myfunc` on the matched AST node.
     */
    class HandleMatch : public MatchFinder::MatchCallback
    {
      std::function<void(const MatchFinder::MatchResult& Result)> handler;
      void run(const MatchFinder::MatchResult& Result) override
      {
        handler(Result);
      }
      ~HandleMatch()
      {
        fprintf(stderr, "HandleMatch destroyed\n");
      }

    public:
      HandleMatch(std::function<void(const MatchFinder::MatchResult& Result)> h)
      : handler(h)
      {}
    };

    /**
     * Generates the pre-compile header into the memory buffer.
     *
     * This method creates a new local Clang just for the pre-compiled headers
     * and returns a memory buffer with the contents, to be inserted in a
     * "file" inside the virtual file system.
     */
    std::unique_ptr<llvm::MemoryBuffer>
    generatePCH(const char* headerFile, SourceLanguage sourceLang)
    {
      Compiler LocalClang(
        llvm::vfs::getRealFileSystem(), headerFile, sourceLang);
      llvm::SmallVector<char, 0> pchOutBuffer;
      auto action = std::make_unique<GenerateMemoryPCHAction>(pchOutBuffer);
      LocalClang.ExecuteAction(*action);
      fprintf(stderr, "PCH is %zu bytes\n", pchOutBuffer.size());
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
      std::string headerFile, SourceLanguage sourceLang = SourceLanguage::CXX)
    : factory(this)
    {
      // Pre-compiles the file requested by the user
      std::unique_ptr<llvm::MemoryBuffer> pchBuffer;
      fprintf(stderr, "\nParsing file %s\n", headerFile.c_str());
      {
        auto t = TimeReport("Computing precompiled headers");
        pchBuffer = generatePCH(headerFile.c_str(), sourceLang);
      }

      // Creating a fake compile unit to include the target file
      // in an in-memory file system.
      fprintf(stderr, "\nCreating fake compile unit\n");
      std::string Code = "#include \"" + headerFile +
        "\"\n"
        "namespace verona { namespace __ffi_internal { \n"
        "}}\n";
      auto Buf = llvm::MemoryBuffer::getMemBufferCopy(Code);
      auto PCHBuf = llvm::MemoryBuffer::getMemBufferCopy(Code);
      FS.addFile(cu_name, std::move(Buf));

      // Adding the pre-compiler header file to the file system.
      auto pchDataRef = llvm::MemoryBuffer::getMemBuffer(
        llvm::MemoryBufferRef{*pchBuffer}, false);
      FS.addFile(headerFile + ".gch", std::move(pchDataRef));

      // Parse the fake compile unit with the user file included inside.
      fprintf(stderr, "\nParsing wrapping unit\n");
      {
        auto t = TimeReport("Creating clang instance");
        // Create the compiler
        Clang = std::make_unique<Compiler>(FS.get(), cu_name, sourceLang);
      }
      auto collectAST = tooling::newFrontendActionFactory(&factory)->create();
      {
        auto t = TimeReport("Reconstructing AST");
        Clang->ExecuteAction(*collectAST);
      }

      // Executing the action consumes the AST.  Reset the compiler instance to
      // refer to the AST that it just parsed and create a Sema instance.
      Clang->setASTMachinery(factory.newASTConsumer(), ast);

      fprintf(stderr, "\nAST: %p\n\n", ast);
    }

    /**
     * Gets an {class | template | enum} type from the source AST by name.
     * The name must exist and be fully qualified and it will match in the
     * order specified above.
     */
    CXXType getType(std::string name)
    {
      name = "::" + name;
      MatchFinder finder;
      const EnumDecl* foundEnum = nullptr;
      const CXXRecordDecl* foundClass = nullptr;
      const ClassTemplateDecl* foundTemplateClass = nullptr;

      finder.addMatcher(
        cxxRecordDecl(hasName(name)).bind("id"),
        new HandleMatch([&](const MatchFinder::MatchResult& Result) {
          auto* decl =
            Result.Nodes.getNodeAs<CXXRecordDecl>("id")->getDefinition();
          if (decl)
          {
            foundClass = decl;
          }
        }));
      finder.addMatcher(
        classTemplateDecl(hasName(name)).bind("id"),
        new HandleMatch([&](const MatchFinder::MatchResult& Result) {
          auto* decl = Result.Nodes.getNodeAs<ClassTemplateDecl>("id");
          if (decl)
          {
            foundTemplateClass = decl;
          }
        }));
      finder.addMatcher(
        enumDecl(hasName(name)).bind("id"),
        new HandleMatch([&](const MatchFinder::MatchResult& Result) {
          auto* decl = Result.Nodes.getNodeAs<EnumDecl>("id");
          if (decl)
          {
            foundEnum = decl;
          }
        }));
      finder.matchAST(*ast);

      // Should onlyt match one, so this is fine.
      if (foundTemplateClass)
      {
        return CXXType(foundTemplateClass);
      }
      if (foundClass)
      {
        return CXXType(foundClass);
      }
      if (foundEnum)
      {
        return CXXType(foundEnum);
      }

      // Return empty type if nothing found.
      return CXXType();
    }

    /// Return the size in bytes of the specified type.
    /// TODO: Move this to CXXType?
    uint64_t getTypeSize(CXXType& t)
    {
      assert(t.kind != CXXType::Kind::Invalid);
      QualType ty;
      if (t.sizeAndAlign.Width == 0)
      {
        switch (t.kind)
        {
          case CXXType::Kind::Invalid:
          case CXXType::Kind::TemplateClass:
            return 0;
          case CXXType::Kind::SpecializedTemplateClass:
          case CXXType::Kind::Class:
            ty = ast->getRecordType(t.getAs<CXXRecordDecl>());
            break;
          case CXXType::Kind::Enum:
            ty = ast->getEnumType(t.getAs<EnumDecl>());
            break;
          case CXXType::Kind::Builtin:
            ty = typeForBuiltin(t.builtTypeKind);
            break;
        }
        t.sizeAndAlign = ast->getTypeInfo(ty);
      }
      return t.sizeAndAlign.Width / 8;
    }

    /// Returns the type as a template argument.
    clang::TemplateArgument createTemplateArgumentForType(CXXType& t)
    {
      switch (t.kind)
      {
        case CXXType::Kind::Invalid:
        case CXXType::Kind::TemplateClass:
          return clang::TemplateArgument{};
        case CXXType::Kind::SpecializedTemplateClass:
        case CXXType::Kind::Class:
          return clang::TemplateArgument{
            ast->getRecordType(t.getAs<CXXRecordDecl>())};
        case CXXType::Kind::Enum:
          return clang::TemplateArgument{ast->getEnumType(t.getAs<EnumDecl>())};
        case CXXType::Kind::Builtin:
          return clang::TemplateArgument{typeForBuiltin(t.builtTypeKind)};
      }
      return nullptr;
    }

    /// Returns the integral literal as a template value.
    /// Floats are returned as empty template arguments.
    clang::TemplateArgument createTemplateArgumentForIntegerValue(
      CXXType::BuiltinTypeKinds ty, uint64_t value)
    {
      if (
        (ty == CXXType::BuiltinTypeKinds::Float) ||
        (ty == CXXType::BuiltinTypeKinds::Double))
      {
        return clang::TemplateArgument{};
      }
      QualType qualTy = typeForBuiltin(ty);
      auto info = ast->getTypeInfo(qualTy);
      llvm::APInt val{static_cast<unsigned int>(info.Width), value};
      auto* literal =
        IntegerLiteral::Create(*ast, val, qualTy, SourceLocation{});
      return TemplateArgument(literal);
    }

    /// Instantiate the class template specialisation if not yet done.
    CXXType instantiateClassTemplate(
      CXXType& classTemplate, llvm::ArrayRef<TemplateArgument> args)
    {
      if (classTemplate.kind != CXXType::Kind::TemplateClass)
      {
        return CXXType{};
      }

      auto& S = Clang->getSema();

      // Check if this specialisation is already present in the AST
      // (declaration, definition, used).
      ClassTemplateDecl* ClassTemplate =
        classTemplate.getAs<ClassTemplateDecl>();
      void* InsertPos = nullptr;
      ClassTemplateSpecializationDecl* Decl =
        ClassTemplate->findSpecialization(args, InsertPos);
      if (!Decl)
      {
        // This is the first time we have referenced this class template
        // specialization. Create the canonical declaration and add it to
        // the set of specializations.
        Decl = ClassTemplateSpecializationDecl::Create(
          *ast,
          ClassTemplate->getTemplatedDecl()->getTagKind(),
          ClassTemplate->getDeclContext(),
          ClassTemplate->getTemplatedDecl()->getBeginLoc(),
          ClassTemplate->getLocation(),
          ClassTemplate,
          args,
          nullptr);
        ClassTemplate->AddSpecialization(Decl, InsertPos);
      }
      // If specialisation hasn't been directly declared yet (by the user),
      // instantiate the declaration.
      if (Decl->getSpecializationKind() == TSK_Undeclared)
      {
        MultiLevelTemplateArgumentList TemplateArgLists;
        TemplateArgLists.addOuterTemplateArguments(args);
        S.InstantiateAttrsForDecl(
          TemplateArgLists, ClassTemplate->getTemplatedDecl(), Decl);
      }
      // If specialisation hasn't been defined yet, create its definition at the
      // end of the file.
      ClassTemplateSpecializationDecl* Def =
        cast_or_null<ClassTemplateSpecializationDecl>(Decl->getDefinition());
      if (!Def)
      {
        SourceLocation InstantiationLoc = Clang->getEndOfFileLocation();
        assert(InstantiationLoc.isValid());
        S.InstantiateClassTemplateSpecialization(
          InstantiationLoc, Decl, TSK_ExplicitInstantiationDefinition);
        Def = cast<ClassTemplateSpecializationDecl>(Decl->getDefinition());
      }
      return CXXType{Def};
    }

    // Exposing some functionality to make this work
    // TODO: Fix the layering issues

    /// Get AST pointer
    const clang::ASTContext* getAST()
    {
      return ast;
    }

  private:
    /// Maps between CXXType and Clang's types.
    /// TODO: Move this to CXXType?
    QualType typeForBuiltin(CXXType::BuiltinTypeKinds ty)
    {
      switch (ty)
      {
        case CXXType::BuiltinTypeKinds::Float:
          return ast->FloatTy;
        case CXXType::BuiltinTypeKinds::Double:
          return ast->DoubleTy;
        case CXXType::BuiltinTypeKinds::Bool:
          return ast->BoolTy;
        case CXXType::BuiltinTypeKinds::SChar:
          return ast->SignedCharTy;
        case CXXType::BuiltinTypeKinds::Char:
          return ast->CharTy;
        case CXXType::BuiltinTypeKinds::UChar:
          return ast->UnsignedCharTy;
        case CXXType::BuiltinTypeKinds::Short:
          return ast->ShortTy;
        case CXXType::BuiltinTypeKinds::UShort:
          return ast->UnsignedShortTy;
        case CXXType::BuiltinTypeKinds::Int:
          return ast->IntTy;
        case CXXType::BuiltinTypeKinds::UInt:
          return ast->UnsignedIntTy;
        case CXXType::BuiltinTypeKinds::Long:
          return ast->LongTy;
        case CXXType::BuiltinTypeKinds::ULong:
          return ast->UnsignedLongTy;
        case CXXType::BuiltinTypeKinds::LongLong:
          return ast->LongLongTy;
        case CXXType::BuiltinTypeKinds::ULongLong:
          return ast->UnsignedLongLongTy;
      }
      // TODO: This is wrong but silences a warning, need to know what's the
      // correct behaviour here.
      return ast->VoidTy;
    }
  };
} // namespace verona::ffi

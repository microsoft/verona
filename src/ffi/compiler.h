// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

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
#include <llvm/Support/VirtualFileSystem.h>

using namespace clang;
namespace
{
  /// Reports time elapsed between creation and destruction.
  ///
  /// Use:
  ///  {
  ///    auto T = TimeReport("My action");
  ///    ... some action ...
  ///  }
  ///  // Here prints "My action: 12ms" to stderr
  class TimeReport
  {
    timespec start;
    std::string name;
#ifdef CLOCK_PROF
    static const clockid_t clock = CLOCK_PROF;
#else
    const clockid_t clock = CLOCK_PROCESS_CPUTIME_ID;
#endif

  public:
    TimeReport(std::string n) : name(n)
    {
      std::atomic_signal_fence(std::memory_order::memory_order_seq_cst);
      clock_gettime(clock, &start);
    }
    ~TimeReport()
    {
      using namespace std::chrono;
      timespec end;
      std::atomic_signal_fence(std::memory_order::memory_order_seq_cst);
      clock_gettime(clock, &end);
      std::atomic_signal_fence(std::memory_order::memory_order_seq_cst);
      auto interval_from_timespec = [](timespec t) {
        return seconds{t.tv_sec} + nanoseconds{t.tv_nsec};
      };
      auto elapsed =
        interval_from_timespec(end) - interval_from_timespec(start);

      fprintf(
        stderr,
        "%s: %ldms\n",
        name.c_str(),
        duration_cast<milliseconds>(elapsed).count());
    }
  };

  using namespace clang::ast_matchers;

  /// Simple handler for indirect dispatch on a Clang AST matcher.
  ///
  /// Use:
  ///  void myfunc(MatchFinder::MatchResult &);
  ///  MatchFinder f;
  ///  f.addMatcher(new HandleMatch(myfunc));
  ///  f.matchAST(*ast);
  ///  // If matches, runs `myfunc` on the matched AST node.
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
}
namespace verona::ffi::compiler
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
    // FIXME: delete public
  public:
    /// Source file name (can be a header, too)
    std::string sourceFile;
    /// The AST root
    clang::ASTContext* ast = nullptr;
    /// Pre-compiled header
    llvm::Optional<clang::PrecompiledPreamble> preamble;

    /// Source languages Clang supports
    enum SourceLanguage
    {
      C,
      CXX,
      ObjC,
      ObjCXX,
      SOURCE_LANGAUGE_ENUM_SIZE
    };
    /// Converts SourceLanguage into string
    static const char* source_language_string(SourceLanguage sl)
    {
      static std::array<const char*, SOURCE_LANGAUGE_ENUM_SIZE> names = {
        "c", "c++", "objective-c", "objective-c++"};
      return names.at(static_cast<int>(sl));
    }
    // std::unique_ptr<CompilerInstance> Clang =
    // std::make_unique<CompilerInstance>();

    /// Simple wrapper for calling clang with arguments on different files.
    class ClangArgs
    {
      /// All arguments, including empty last one that is replaced evey call
      std::vector<const char*> args;
      /// Position of last argument (filename)
      size_t file_pos;

    public:
      /// C-tor, initialises default arguments + file pos
      ClangArgs(SourceLanguage sourceLang = CXX)
      {
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
        file_pos = args.size() - 1;
        fprintf(stderr, "Clang args created\n");
      }
      /// Returns the array with the filename as the last arg
      llvm::ArrayRef<const char*> getArgs(const char* filename)
      {
        fprintf(stderr, "Clang called for %s\n", filename);
        args[file_pos] = filename;
        return args;
      }
    };

    /// Name of the internal compilation unit that includes the filename
    static constexpr const char* cu_name = "verona_interface.cc";

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
     * Creates an in-memory overlay file-system, so we can create the interim
     * compile unit (that includes the user file) alongside the necessary
     * headers to include (built-in, etc).
     */
    std::pair<
      IntrusiveRefCntPtr<llvm::vfs::FileSystem>,
      IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem>>
    createOverlayFilesystem(std::unique_ptr<llvm::MemoryBuffer> Buf)
    {
      IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> inMemoryVFS =
        new llvm::vfs::InMemoryFileSystem();
      IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> Overlay =
        new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem());
      Overlay->pushOverlay(inMemoryVFS);
      inMemoryVFS->addFile(cu_name, time(nullptr), std::move(Buf));
      // FIXME: We should also add Clang's built-in headers.
      return {IntrusiveRefCntPtr<llvm::vfs::FileSystem>{Overlay}, inMemoryVFS};
    }

    /// Compiler instance.
      std::unique_ptr<CompilerInstance> Clang;

    /**
     * Creates the Clang instance, with preprocessor and header search support.
     */
    void createClangInstance(
      llvm::ArrayRef<const char*> args,
      llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS)
    {
      Clang = std::make_unique<CompilerInstance>();
      // TODO: Wire up diagnostics so that we can spot invalid template
      // instantiations.
      IntrusiveRefCntPtr<DiagnosticIDs> DiagID = new DiagnosticIDs();
      IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
      auto* DiagsPrinter = new TextDiagnosticPrinter{llvm::errs(), &*DiagOpts};
      auto* Diags =
        new DiagnosticsEngine(DiagID, DiagOpts, DiagsPrinter, false);
      fprintf(stderr, "Diags: %p\n", Diags);
      auto CI = createInvocationFromCommandLine(
        args, Diags, llvm::vfs::getRealFileSystem());
      Diags = new DiagnosticsEngine(DiagID, DiagOpts, DiagsPrinter, false);
      fprintf(stderr, "CI: %p\n", CI.get());
      fprintf(stderr, "Clang: %p\n", Clang.get());
      auto* fileManager = new FileManager(FileSystemOptions{}, VFS);
      auto* sourceMgr = new SourceManager(
        *Diags,
        *fileManager,
        /*UserFilesAreVolatile*/ false);
      Clang->setFileManager(fileManager);
      Clang->setSourceManager(sourceMgr);
      Clang->setInvocation(std::move(CI));
      Clang->setDiagnostics(Diags);
      auto PPOpts = std::make_shared<PreprocessorOptions>();
      TrivialModuleLoader TML;
      auto HeaderSearchPtr = std::make_unique<HeaderSearch>(
        std::make_shared<HeaderSearchOptions>(),
        *sourceMgr,
        *Diags,
        Clang->getLangOpts(),
        nullptr);
      auto PreprocessorPtr = std::make_shared<Preprocessor>(
        PPOpts,
        *Diags,
        Clang->getLangOpts(),
        *sourceMgr,
        *HeaderSearchPtr,
        TML,
        nullptr,
        false);
      Clang->setPreprocessor(PreprocessorPtr);
      Clang->getPreprocessor().enableIncrementalProcessing();
      // FIXME: Do something more sensible with the diagnostics engine so
      // that we can propagate errors to Verona
    }

    /// Pre-compiled memory buffer.
    std::unique_ptr<llvm::MemoryBuffer> pchBuffer;

    /// Generates the pre-compile header into the memory buffer.
    std::unique_ptr<llvm::MemoryBuffer>
    generatePCH(std::string headerFile, ArrayRef<const char*> args)
    {
      createClangInstance(args, llvm::vfs::getRealFileSystem());
      llvm::SmallVector<char, 0> pchOutBuffer;
      auto action = std::make_unique<GenerateMemoryPCHAction>(pchOutBuffer);
      Clang->ExecuteAction(*action);
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
    CXXInterface(std::string headerFile, SourceLanguage sourceLang = CXX)
    : sourceFile(headerFile), factory(this)
    {
      ClangArgs args(sourceLang);

      // Pre-compiles the file requested by the user
      fprintf(stderr, "\nParsing file\n");
      {
        auto t = TimeReport("Computing precompiled headers");
        pchBuffer = generatePCH(headerFile, args.getArgs(headerFile.c_str()));
      }

      // Creating a fake compile unit to include the target file
      // in an in-memory file system.
      fprintf(stderr, "\nCreating in-memory file system\n");
      std::string Code = "#include \"" + headerFile +
        "\"\n"
        "namespace verona { namespace __ffi_internal { \n"
        "}}\n";
      auto Buf = llvm::MemoryBuffer::getMemBufferCopy(Code);
      auto PCHBuf = llvm::MemoryBuffer::getMemBufferCopy(Code);
      auto [VFS, inMemoryVFS] = createOverlayFilesystem(std::move(Buf));

      // Adding the pre-compiler header file to the file system.
      auto pchDataRef = llvm::MemoryBuffer::getMemBuffer(
        llvm::MemoryBufferRef{*pchBuffer}, false);
      inMemoryVFS->addFile(
        headerFile + ".gch", time(nullptr), std::move(pchDataRef));

      // Parse the fake compile unit with the user file included inside.
      fprintf(stderr, "\nParsing wrapping unit\n");
      {
        auto t = TimeReport("Creating clang instance");
        createClangInstance(args.getArgs(cu_name), VFS);
      }
      auto collectAST = tooling::newFrontendActionFactory(&factory)->create();
      {
        auto t = TimeReport("Reconstructing AST");
        Clang->ExecuteAction(*collectAST);
      }

      // Executing the action consumes the AST.  Reset the compiler instance to
      // refer to the AST that it just parsed and create a Sema instance.
      Clang->setASTConsumer(factory.newASTConsumer());
      Clang->setASTContext(ast);
      Clang->createSema(TU_Complete, nullptr);

      fprintf(stderr, "\nAST: %p\n\n", ast);
    }

    /// LLVMContext for LLVM lowering.
    std::unique_ptr<llvm::LLVMContext> llvmContext{new llvm::LLVMContext};

    /**
     * Lowers each top-level declaration to LLVM IR and dumps the module.
     */
    std::unique_ptr<llvm::Module> emitLLVM()
    {
      std::unique_ptr<CodeGenerator> CodeGen{CreateLLVMCodeGen(
        Clang->getDiagnostics(),
        cu_name,
        Clang->getHeaderSearchOpts(),
        Clang->getPreprocessorOpts(),
        Clang->getCodeGenOpts(),
        *llvmContext)};
      fprintf(stderr, "Generating LLVM IR...\n");
      CodeGen->Initialize(*ast);
      CodeGen->HandleTranslationUnit(*ast);
      for (auto& D : ast->getTranslationUnitDecl()->decls())
        CodeGen->HandleTopLevelDecl(DeclGroupRef{D});
      std::unique_ptr<llvm::Module> M{CodeGen->ReleaseModule()};
      fprintf(stderr, "M: %p\n", M.get());
      M->dump();
      // Note: `M` must be freed before `this`
      return M;
    }

    /**
     * C++ types that can be queried from the AST matchers.
     */
    struct CXXType
    {
      /// Match kinds.
      enum class Kind
      {
        Invalid,
        TemplateClass,
        SpecializedTemplateClass,
        Class,
        Enum,
        Builtin
      } kind = Kind::Invalid;

      /// C++ builtin types.
      enum class BuiltinTypeKinds
      {
        Bool,
        SChar,
        Char,
        UChar,
        Short,
        UShort,
        Int,
        UInt,
        Long,
        ULong,
        LongLong,
        ULongLong,
        Float,
        Double
      };

      /// Converts kind name to string.
      const char* kindName()
      {
        switch (kind)
        {
          case Kind::Invalid:
            return "Invalid";
          case Kind::SpecializedTemplateClass:
            return "Specialized Class Template";
          case Kind::TemplateClass:
            return "Class Template";
          case Kind::Class:
            return "Class";
          case Kind::Enum:
            return "Enum";
          case Kind::Builtin:
            return "Builtin";
        }
        return nullptr;
      }

      /// Returns true if the type is templated.
      bool isTemplate()
      {
        return kind == Kind::TemplateClass;
      }

      /// CXXType builtin c-tor
      CXXType(BuiltinTypeKinds t)
      : kind(Kind::Builtin), decl(nullptr), builtTypeKind(t)
      {}
      /// CXXType class c-tor
      CXXType(const CXXRecordDecl* d) : kind(Kind::Class), decl(d) {}
      /// CXXType template class c-tor
      CXXType(const ClassTemplateDecl* d) : kind(Kind::TemplateClass), decl(d)
      {}
      /// CXXType template specialisation class c-tor
      CXXType(const ClassTemplateSpecializationDecl* d)
      : kind(Kind::SpecializedTemplateClass), decl(d)
      {}
      /// CXXType enum c-tor
      CXXType(const EnumDecl* d) : kind(Kind::Enum), decl(d) {}
      /// CXXType empty c-tor (Invalid)
      CXXType() = default;
      /// Returns the number of template parameter, if class is a template.
      int numberOfTemplateParameters()
      {
        if (!isTemplate())
        {
          return 0;
        }
        return getAs<ClassTemplateDecl>()->getTemplateParameters()->size();
      }

      // private:

      /**
       * Access the underlying decl as the specified type.  This removes the
       * `const` qualification, allowing the AST to be modified, and should be
       * used only by the Clang AST interface classes.
       */
      template<class T>
      T* getAs()
      {
        return dyn_cast<T>(const_cast<NamedDecl*>(decl));
      }
      /**
       * The declaration that corresponds to this type.
       */
      const NamedDecl* decl = nullptr;
      /**
       * The kind if this is a builtin.
       * FIXME: This and `decl` should be a union, only one is ever present at
       * a time.
       */
      BuiltinTypeKinds builtTypeKind;
      /**
       * The size and alignment of this type.  Note that this is not valid for
       * templated types that are not fully specified.
       */
      clang::TypeInfo sizeAndAlign;
    };

    /// Returns the CXXType if a builtin type.
    /// FIXME: Should this be static?
    CXXType getBuiltinType(CXXType::BuiltinTypeKinds k)
    {
      return CXXType{k};
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
        auto& SM = Clang->getSourceManager();
        auto mainFile = SM.getMainFileID();
        SourceLocation InstantiationLoc = SM.getLocForEndOfFile(mainFile);
        assert(InstantiationLoc.isValid());
        S.InstantiateClassTemplateSpecialization(
          InstantiationLoc, Decl, TSK_ExplicitInstantiationDefinition);
        Def = cast<ClassTemplateSpecializationDecl>(Decl->getDefinition());
      }
      return CXXType{Def};
    }

  private:
    /// Maps between CXXType and Clang's types.
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
} // namespace verona::ffi::compiler

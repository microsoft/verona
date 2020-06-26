// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

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
#include <stdio.h>

using namespace clang;
namespace
{
  class TimeReport
  {
    timespec start;
    std::string name;
    static const clockid_t clock = CLOCK_PROF;

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
        "%s: %lldms\n",
        name.c_str(),
        duration_cast<milliseconds>(elapsed).count());
    }
  };

  using namespace clang::ast_matchers;

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

  struct GenerateMemoryPCHAction : GeneratePCHAction
  {
    llvm::SmallVectorImpl<char>& vec;
    GenerateMemoryPCHAction(llvm::SmallVectorImpl<char>& vec) : vec(vec) {}
    std::unique_ptr<ASTConsumer>
    CreateASTConsumer(CompilerInstance& CI, StringRef InFile)
    {
      std::string Sysroot;
      if (!ComputeASTConsumerArguments(CI, /*ref*/ Sysroot))
      {
        return nullptr;
      }

      std::string OutputFile;
      auto OS = std::make_unique<llvm::raw_svector_ostream>(vec);
      const auto& FrontendOpts = CI.getFrontendOpts();
      auto Buffer = std::make_shared<PCHBuffer>();
      std::vector<std::unique_ptr<ASTConsumer>> Consumers;
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
      Consumers.push_back(
        CI.getPCHContainerWriter().CreatePCHContainerGenerator(
          CI, InFile.str(), OutputFile, std::move(OS), Buffer));

      return std::make_unique<MultiplexConsumer>(std::move(Consumers));
    }
  };
}

namespace verona
{
  class CXXInterface
  {
    // FIXME: delete public
  public:
    std::string sourceFile;
    clang::ASTContext* ast = nullptr;
    llvm::Optional<clang::PrecompiledPreamble> preamble;

    enum SourceLanguage
    {
      C,
      CXX,
      ObjC,
      ObjCXX,
      SOURCE_LANGAUGE_ENUM_SIZE
    };
    static const char* source_language_string(SourceLanguage sl)
    {
      static std::array<const char*, SOURCE_LANGAUGE_ENUM_SIZE> names = {
        "c", "c++", "objective-c", "objective-c++"};
      return names.at(static_cast<int>(sl));
    }
    // std::unique_ptr<CompilerInstance> Clang =
    // std::make_unique<CompilerInstance>();

    static constexpr const char* cu_name = "verona_interface.cc";

    struct ASTConsumerFactory
    {
      CXXInterface* consumer;
      struct Collector : public clang::ASTConsumer
      {
        CXXInterface* consumer;
        Collector(CXXInterface* c) : consumer(c) {}
        void HandleTranslationUnit(ASTContext& Ctx) override
        {
          fprintf(stderr, "AST consumer %p received AST %p\n", this, &Ctx);
          consumer->ast = &Ctx;
        }
        ~Collector()
        {
          fprintf(stderr, "AST consumer %p destroyed\n", this);
        }
      };
      ASTConsumerFactory(CXXInterface* c) : consumer(c) {}
      std::unique_ptr<clang::ASTConsumer> newASTConsumer()
      {
        return std::make_unique<Collector>(consumer);
      }
    } factory;

    friend class ASTConsumerFactory::Collector;

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

    std::unique_ptr<llvm::MemoryBuffer> pchBuffer;
    struct CompilerState
    {
      std::shared_ptr<Preprocessor> Preprocessor;
      std::unique_ptr<HeaderSearch> HeaderSearch;
      std::unique_ptr<CompilerInstance> Clang;
    };
    std::unique_ptr<CompilerState> queryCompilerState;

    std::unique_ptr<CompilerState> createClangInstance(
      llvm::ArrayRef<const char*> args,
      llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS)
    {
      auto State = std::make_unique<CompilerState>();
      State->Clang = std::make_unique<CompilerInstance>();
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
      fprintf(stderr, "Clang: %p\n", State->Clang.get());
      auto* fileManager = new FileManager(FileSystemOptions{}, VFS);
      auto* sourceMgr = new SourceManager(
        *Diags,
        *fileManager,
        /*UserFilesAreVolatile*/ false);
      State->Clang->setFileManager(fileManager);
      State->Clang->setSourceManager(sourceMgr);
      State->Clang->setInvocation(std::move(CI));
      State->Clang->setDiagnostics(Diags);
      auto PPOpts = std::make_shared<PreprocessorOptions>();
      TrivialModuleLoader TML;
      State->HeaderSearch = std::make_unique<HeaderSearch>(
        std::make_shared<HeaderSearchOptions>(),
        *sourceMgr,
        *Diags,
        State->Clang->getLangOpts(),
        nullptr);
      State->Preprocessor = std::make_shared<Preprocessor>(
        PPOpts,
        *Diags,
        State->Clang->getLangOpts(),
        *sourceMgr,
        *State->HeaderSearch,
        TML,
        nullptr,
        false);
      State->Clang->setPreprocessor(State->Preprocessor);
      State->Clang->getPreprocessor().enableIncrementalProcessing();
      // FIXME: Do something more sensible with the diagnostics engine so
      // that we can propagate errors to Verona
      return State;
    }

    std::unique_ptr<llvm::MemoryBuffer>
    generatePCH(std::string headerFile, ArrayRef<const char*> args)
    {
      auto pchCompilerState =
        createClangInstance(args, llvm::vfs::getRealFileSystem());
      llvm::SmallVector<char, 0> pchOutBuffer;
      auto action = std::make_unique<GenerateMemoryPCHAction>(pchOutBuffer);
      pchCompilerState->Clang->ExecuteAction(*action);
      fprintf(stderr, "PCH is %zu bytes\n", pchOutBuffer.size());
      return std::unique_ptr<llvm::MemoryBuffer>(
        new llvm::SmallVectorMemoryBuffer(std::move(pchOutBuffer)));
    }

  public:
    CXXInterface(std::string headerFile, SourceLanguage sourceLang = CXX)
    : sourceFile(headerFile), factory(this)
    {
      const char* langName = source_language_string(sourceLang);
      // FIXME: Don't hard code include paths!
      const char* pchArgs[] = {"clang",
                               "-x",
                               langName,
                               "-I",
                               "/usr/local/llvm80/include/",
                               headerFile.c_str()};

      fprintf(stderr, "Computing precompiled preamble\n");
      {
        auto t = TimeReport("Building PCH");
        pchBuffer = generatePCH(headerFile, pchArgs);
      }

      fprintf(stderr, "Parsing fake CU\n");
      const char* args[] = {
        "clang", "-x", langName, "-I", "/usr/local/llvm80/include/", cu_name};
      std::string Code = "#include <" + headerFile +
        ">\n"
        "namespace verona { namespace __ffi_internal { \n"
        "}}\n";
      auto Buf = llvm::MemoryBuffer::getMemBufferCopy(Code);
      auto PCHBuf = llvm::MemoryBuffer::getMemBufferCopy(Code);
      auto [VFS, inMemoryVFS] = createOverlayFilesystem(std::move(Buf));

      auto pchDataRef = llvm::MemoryBuffer::getMemBuffer(
        llvm::MemoryBufferRef{*pchBuffer}, false);
      inMemoryVFS->addFile(
        headerFile + ".gch", time(nullptr), std::move(pchDataRef));

      queryCompilerState = createClangInstance(args, VFS);
      auto collectAST = tooling::newFrontendActionFactory(&factory)->create();
      fprintf(stderr, "Parsing stub (no AST: %p)\n", ast);
      {
        auto t = TimeReport("Reconstructing AST");
        queryCompilerState->Clang->ExecuteAction(*collectAST);
      }
      // Executing the action consumes the AST.  Reset the compiler instance to
      // refer to the AST that it just parsed and create a Sema instance.
      queryCompilerState->Clang->setASTConsumer(factory.newASTConsumer());
      queryCompilerState->Clang->setASTContext(ast);
      queryCompilerState->Clang->createSema(TU_Complete, nullptr);
      fprintf(stderr, "AST: %p\n", ast);
    }

    std::unique_ptr<llvm::LLVMContext> llvmContext{new llvm::LLVMContext};

    std::unique_ptr<llvm::Module> emitLLVM()
    {
      auto& CI = queryCompilerState->Clang;
      std::unique_ptr<CodeGenerator> CodeGen{CreateLLVMCodeGen(
        CI->getDiagnostics(),
        cu_name,
        CI->getHeaderSearchOpts(),
        CI->getPreprocessorOpts(),
        CI->getCodeGenOpts(),
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

    struct CXXType
    {
      enum class Kind
      {
        Invalid,
        TemplateClass,
		SpecializedTemplateClass,
        Class,
        Enum,
        Builtin
      } kind = Kind::Invalid;

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
      }

      bool isTemplate()
      {
        return kind == Kind::TemplateClass;
      }

      CXXType(BuiltinTypeKinds t)
      : kind(Kind::Builtin), decl(nullptr), builtTypeKind(t)
      {}
      CXXType(const CXXRecordDecl* d) : kind(Kind::Class), decl(d) {}
      CXXType(const ClassTemplateDecl* d) : kind(Kind::TemplateClass), decl(d)
      {}
      CXXType(const ClassTemplateSpecializationDecl* d) : kind(Kind::SpecializedTemplateClass), decl(d)
      {}
      CXXType(const EnumDecl* d) : kind(Kind::Enum), decl(d) {}
      CXXType() = default;
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

    CXXType getBuiltinType(CXXType::BuiltinTypeKinds k)
    {
      return CXXType{k};
    }
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
      return CXXType();
    }

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
    }

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
    CXXType instantiateClassTemplate(
      CXXType& classTemplate, llvm::ArrayRef<TemplateArgument> args)
    {
      if (classTemplate.kind != CXXType::Kind::TemplateClass)
      {
        return CXXType{};
      }

      auto& S = queryCompilerState->Clang->getSema();

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
      if (Decl->getSpecializationKind() == TSK_Undeclared)
      {
        MultiLevelTemplateArgumentList TemplateArgLists;
        TemplateArgLists.addOuterTemplateArguments(args);
        S.InstantiateAttrsForDecl(
          TemplateArgLists, ClassTemplate->getTemplatedDecl(), Decl);
      }
      ClassTemplateSpecializationDecl* Def =
        cast_or_null<ClassTemplateSpecializationDecl>(Decl->getDefinition());
      if (!Def)
      {
        auto& SM = queryCompilerState->Clang->getSourceManager();
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
    }
  };
};

using namespace verona;

int main(void)
{
  using CXXType = CXXInterface::CXXType;

  auto test = [](auto& interface, const char* name) {
    auto t = TimeReport(std::string("Looking up") + name);
    auto decl = interface.getType(name);
    auto* d = decl.decl;
    if (decl.kind != CXXType::Kind::Invalid)
    {
      fprintf(
        stderr,
        "Found: (%p) %s : %s\n",
        d,
        decl.kindName(),
        d->getName().str().c_str());
    }
    else
    {
      fprintf(stderr, "Not found: %s\n", name);
    }
  };
	// FIXME: Verona compiler should be able to find the path and pass include
	// paths to this interface.
	CXXInterface interface("/usr/local/llvm80/include/llvm/IR/IRBuilder.h");

	test(interface, "llvm::Value");
	test(interface, "llvm::Type::TypeID");
	test(interface, "llvm::IRBuilder");
	auto irb = interface.getType("llvm::IRBuilder");
	auto params = dyn_cast<ClassTemplateDecl>(irb.decl)->getTemplateParameters();
	std::vector<TemplateArgument> args;
	for (auto param : *params)
	{
		if (auto typeParam = dyn_cast<TemplateTypeParmDecl>(param))
		{
			args.push_back(typeParam->getDefaultArgument());
		}
		else if (auto nontypeParam = dyn_cast<NonTypeTemplateParmDecl>(param))
		{
			args.push_back(nontypeParam->getDefaultArgument());
		}
	}
	TemplateName irbName{dyn_cast<TemplateDecl>(const_cast<NamedDecl*>(irb.decl))};
	interface.ast->getTemplateSpecializationType(irbName, args).dump();

  CXXInterface stdarray("/usr/include/c++/v1/array");
  auto t = TimeReport("Instantiating std::array");
  auto arr = stdarray.getType("std::array");
  fprintf(
    stderr,
    "std::array has %d template parameters\n",
    arr.numberOfTemplateParameters());
  auto IntTy = stdarray.getBuiltinType(CXXType::BuiltinTypeKinds::Int);
  auto TypeArg = stdarray.createTemplateArgumentForType(IntTy);
  auto ValueArg = stdarray.createTemplateArgumentForIntegerValue(
    CXXType::BuiltinTypeKinds::Int, 4);

  CXXType arrIntFour = stdarray.instantiateClassTemplate(arr, {TypeArg, ValueArg});
  arr.decl->dump();
  arrIntFour.decl->dump();
  fprintf(stderr, "std::array<int, 4> is %lu bytes\n", stdarray.getTypeSize(arrIntFour));

  auto* DC = stdarray.ast->getTranslationUnitDecl();
  auto& SM = stdarray.queryCompilerState->Clang->getSourceManager();
  auto mainFile = SM.getMainFileID();
  SourceLocation loc = SM.getLocForEndOfFile(mainFile);
  IdentifierInfo& fnNameIdent = stdarray.ast->Idents.get("verona_wrapper_fn_1");
  DeclarationName fnName{&fnNameIdent};
  QualType retTy = stdarray.ast->IntTy;
  QualType argTy = stdarray.ast->IntTy;
  FunctionProtoType::ExtProtoInfo EPI;
  QualType fnTy = stdarray.ast->getFunctionType(retTy, {argTy}, EPI);

  auto newFunction = FunctionDecl::Create(
    *stdarray.ast,
    DC,
    loc,
    loc,
    fnName,
    fnTy,
    stdarray.ast->getTrivialTypeSourceInfo(fnTy),
    StorageClass::SC_None);
  newFunction->setLexicalDeclContext(DC);

  IdentifierInfo& arg1Ident = stdarray.ast->Idents.get("arg1");
  ParmVarDecl* arg1Decl = ParmVarDecl::Create(
    *stdarray.ast,
    newFunction,
    loc,
    loc,
    &arg1Ident,
    argTy,
    nullptr,
    StorageClass::SC_None,
    nullptr);
  newFunction->setParams({arg1Decl});

  llvm::APInt four{32, 4};
  auto* fourLiteral = IntegerLiteral::Create(
    *stdarray.ast, four, stdarray.ast->IntTy, SourceLocation{});
  auto retStmt = ReturnStmt::Create(*stdarray.ast, loc, fourLiteral, nullptr);
  newFunction->setBody(retStmt);

  newFunction->dump();
  DC->addDecl(newFunction);

  // DC->dump();

  // Decl->dump();
  // Def->dump();

  stdarray.emitLLVM();

  TemplateName arrName{arr.getAs<TemplateDecl>()};
  stdarray.ast
    ->getCanonicalTemplateSpecializationType(arrName, {TypeArg, ValueArg})
    .dump();
  stdarray.ast
    ->getCanonicalTemplateSpecializationType(arrName, {TypeArg, TypeArg})
    .dump();

}

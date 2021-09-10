// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "CXXType.h"
#include "Compiler.h"

#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/Sema/Sema.h>
#include <clang/Sema/Template.h>
#include <clang/Sema/TemplateDeduction.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>

// Makes matcher syntax so much clearer
using namespace clang::ast_matchers;

namespace verona::interop
{
  class CXXInterface;

  /**
   * C++ Clang Query.
   *
   * This class implements the queries that users of CXXInterface will do to
   * find types, functions, template arguments, etc.
   *
   * As the Verona compiler goes through type inference, it needs to know
   * what foreign types are used, what foreign functions are called, etc.
   *
   * The API only checks for validity and returns the equivalent of a
   * declaration, not an implementation. None of the functions and type
   * constructors are actually implemented. The CXXImpl class will handle
   * that at code generation phase, from the declarations created here.
   */
  class CXXQuery
  {
    /// The AST root
    clang::ASTContext* ast = nullptr;

    /**
     * Simple handler for indirect dispatch on a CXXType AST matcher.
     *
     * Use:
     * ```
     *  MatchFinder f;
     *  f.addMatcher(new CXXTypeMatch<clang::CXXRecordDecl>(cxxTy));
     *  f.matchAST(*ast);
     *  // If matches, puts result in `cxxTy`
     * ```
     */
    template<class DeclTy>
    class CXXTypeMatch : public MatchFinder::MatchCallback
    {
      /// Type store owned by caller
      CXXType& store;

      /// Store the match on the caller's CXXType if empty
      void run(const MatchFinder::MatchResult& Result) override
      {
        auto* decl = Result.Nodes.getNodeAs<DeclTy>("id");
        // Only store the first match (FIXME?)
        if (decl && !store.valid())
        {
          store = CXXType(decl);
        }
      }

    public:
      CXXTypeMatch(CXXType& store) : store(store) {}
    };

    /**
     * Maps between CXXType and Clang's types.
     */
    clang::QualType typeForBuiltin(CXXType::BuiltinTypeKinds ty) const
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

    /**
     * Returns the type as a template argument.
     */
    clang::TemplateArgument getTemplateArgument(CXXType t) const
    {
      switch (t.kind)
      {
        case CXXType::Kind::Invalid:
        case CXXType::Kind::TemplateClass:
          // TODO: Fix template class
          return clang::TemplateArgument{};
        case CXXType::Kind::SpecializedTemplateClass:
        case CXXType::Kind::Class:
          return clang::TemplateArgument{
            ast->getRecordType(t.getAs<clang::CXXRecordDecl>())};
        case CXXType::Kind::Enum:
          return clang::TemplateArgument{
            ast->getEnumType(t.getAs<clang::EnumDecl>())};
        case CXXType::Kind::Builtin:
          return clang::TemplateArgument{typeForBuiltin(t.builtTypeKind)};
      }
      return nullptr;
    }

    /**
     * Returns the integral literal as a template argument.
     * TODO: C++20 accepts floating point too
     */
    clang::TemplateArgument
    getTemplateArgument(CXXType::BuiltinTypeKinds ty, uint64_t value) const
    {
      assert(CXXType::isIntegral(ty));
      clang::QualType qualTy = typeForBuiltin(ty);
      auto info = ast->getTypeInfo(qualTy);
      llvm::APInt val{static_cast<unsigned int>(info.Width), value};
      auto* literal = clang::IntegerLiteral::Create(
        *ast, val, qualTy, clang::SourceLocation{});
      return clang::TemplateArgument(literal);
    }

  public:
    /**
     * CXXQuery c-tor. Creates the internal compile unit, include the
     * user file (and all dependencies), generates the pre-compiled headers,
     * creates the compiler instance and re-attaches the AST to the interface.
     */
    CXXQuery(clang::ASTContext* ast, Compiler* Clang) : ast(ast) {}

    /**
     * Gets an {class | template | enum} type from the source AST by name.
     * The name must exist and be fully qualified and it will match in the
     * order specified above.
     *
     * We don't need to find builtin types because they're pre-defined in the
     * language and represented in CXXType directly.
     *
     * TODO: Change this method to receive a list of names and return a list
     * of types (or some variation over mutliple types at the same time).
     */
    CXXType getType(std::string name) const
    {
      // Check for builtins first to avoid a trip down the AST
      CXXType ty = llvm::StringSwitch<CXXType>(name)
                     .Case("bool", CXXType::getBoolean())
                     .Case("unsigned char", CXXType::getUnsignedChar())
                     .Case("char", CXXType::getChar())
                     .Case("signed char", CXXType::getSignedChar())
                     .Case("short", CXXType::getShort())
                     .Case("unsigned short", CXXType::getUnsignedShort())
                     .Case("int", CXXType::getInt())
                     .Case("unsigned int", CXXType::getUnsignedInt())
                     .Case("long", CXXType::getLong())
                     .Case("unsigned long", CXXType::getUnsignedLong())
                     .Case("long long", CXXType::getLongLong())
                     .Case("unsigned long long", CXXType::getUnsignedLongLong())
                     .Case("float", CXXType::getFloat())
                     .Case("double", CXXType::getDouble())
                     .Default(CXXType());

      // If type is builtin, early return
      if (ty.valid())
        return ty;

      // Search for class, enum or template.
      name = "::" + name;
      MatchFinder finder;
      auto recDeclMatch =
        std::make_unique<CXXTypeMatch<clang::CXXRecordDecl>>(ty);
      auto classTempMatch =
        std::make_unique<CXXTypeMatch<clang::ClassTemplateDecl>>(ty);
      auto enumDeclMatch = std::make_unique<CXXTypeMatch<clang::EnumDecl>>(ty);

      finder.addMatcher(
        cxxRecordDecl(hasName(name)).bind("id"), recDeclMatch.get());
      finder.addMatcher(
        classTemplateDecl(hasName(name)).bind("id"), classTempMatch.get());
      finder.addMatcher(
        enumDecl(hasName(name)).bind("id"), enumDeclMatch.get());
      finder.matchAST(*ast);

      // Return the type matched directly.
      // If there was no match, the type is still invalid
      return ty;
    }

    /**
     * Return the size in bytes of the specified type
     */
    uint64_t getTypeSize(CXXType& t) const
    {
      assert(t.kind != CXXType::Kind::Invalid);
      if (t.sizeAndAlign.Width == 0)
      {
        clang::QualType ty = getQualType(t);
        t.sizeAndAlign = ast->getTypeInfo(ty);
      }
      return t.sizeAndAlign.Width / 8;
    }

    /**
     * Return the qualified type for a CXXType
     * CXXBuilder uses this for creating AST nodes
     */
    clang::QualType getQualType(CXXType ty) const
    {
      switch (ty.kind)
      {
        case CXXType::Kind::Invalid:
        case CXXType::Kind::TemplateClass:
          // TODO: Fix template class
          return clang::QualType{};
        case CXXType::Kind::SpecializedTemplateClass:
        case CXXType::Kind::Class:
          return ast->getRecordType(ty.getAs<clang::CXXRecordDecl>());
        case CXXType::Kind::Enum:
          return ast->getEnumType(ty.getAs<clang::EnumDecl>());
        case CXXType::Kind::Builtin:
          return typeForBuiltin(ty.builtTypeKind);
      }
      // TODO: This is wrong but silences a warning, need to know what's the
      // correct behaviour here.
      return ast->VoidTy;
    }

    /**
     * Get field by name from a (template) class type. Returns nullptr if
     * the field doesn't exist or type isn't class/struct.
     *
     * FIXME: Do we also need to pass the type?
     */
    clang::FieldDecl* getField(CXXType& ty, llvm::StringRef name) const
    {
      // Type must be a class (TODO: should this be an assert?)
      if (!ty.isClass())
        return nullptr;

      // Compare the name, maybe we need to compare the types, too?
      auto decl = ty.getAs<clang::CXXRecordDecl>();
      for (auto field : decl->fields())
      {
        if (field->getName() == name)
          return field;
      }

      // We couldn't find anything
      return nullptr;
    }

    /**
     * Get function by name and signature from a (template) class type.
     * Returns nullptr if the function doesn't exist or type isn't class/struct.
     */
    clang::CXXMethodDecl* getMethod(
      CXXType& ty, llvm::StringRef name, llvm::ArrayRef<std::string> args) const
    {
      // Type must be a class (TODO: should this be an assert?)
      if (!ty.isClass())
        return nullptr;

      auto decl = ty.getAs<clang::CXXRecordDecl>();
      for (auto m : decl->methods())
      {
        auto func = llvm::dyn_cast<clang::FunctionDecl>(m);
        // First, compare the function name, must be the same
        if (func->getName() != name)
          continue;

        // Quick fail for number of arguments
        if (args.size() != func->parameters().size())
          continue;

        // Then make sure the argument types are the same, too
        auto compare =
          [this](const clang::ParmVarDecl* parm, const std::string& str) {
            auto parmTy = parm->getOriginalType()->getPointeeType();
            auto argTy = getQualType(getType(str))->getPointeeType();
            return parmTy == argTy;
          };
        if (std::equal(
              func->param_begin(), func->param_end(), args.begin(), compare))
        {
          return m;
        }

        // Otherwise, just continue and try the next
      }

      // We couldn't find anything
      return nullptr;
    }

    /**
     * Create the parameters of a template class from type names or values
     */
    llvm::SmallVector<clang::TemplateArgument, 1>
    gatherTemplateArguments(CXXType& ty, llvm::ArrayRef<std::string> args) const
    {
      llvm::SmallVector<clang::TemplateArgument, 1> templateArgs;

      // First, detect all user declared arguments, overriding default
      // arguments.
      for (auto arg : args)
      {
        if (isdigit(arg[0]))
        {
          // Numbers default to int parameter
          auto num = atol(arg.c_str());
          templateArgs.push_back(
            getTemplateArgument(CXXType::BuiltinTypeKinds::Int, num));
        }
        else
        {
          // Try to find the type name
          auto decl = getType(arg);
          assert(decl.valid());
          templateArgs.push_back(getTemplateArgument(decl));
        }
      }

      // If there are any remaining arguments, get their default values and add
      // to the list. We want to create all template classes fully defined, to
      // make sure there are no dependent types left.
      auto actual = llvm::dyn_cast<clang::ClassTemplateDecl>(ty.decl)
                      ->getTemplateParameters();
      auto skip = args.size();
      auto all = actual->size();
      for (auto i = skip; i < all; i++)
      {
        auto param = actual->getParam(i);
        // Type parameter
        if (auto typeParam = llvm::dyn_cast<clang::TemplateTypeParmDecl>(param))
        {
          assert(typeParam->hasDefaultArgument());
          templateArgs.push_back(typeParam->getDefaultArgument());
        }
        // Non-type parameter
        else if (
          auto nontypeParam =
            llvm::dyn_cast<clang::NonTypeTemplateParmDecl>(param))
        {
          assert(typeParam->hasDefaultArgument());
          templateArgs.push_back(nontypeParam->getDefaultArgument());
        }
      }

      // All args, passed and default, of the template declaration
      return templateArgs;
    }

    /**
     * Get the canonical template especialization with args.
     */
    clang::QualType getCanonicalTemplateSpecializationType(
      const clang::NamedDecl* decl,
      llvm::ArrayRef<clang::TemplateArgument> args) const
    {
      clang::TemplateName templ{clang::dyn_cast<clang::TemplateDecl>(
        const_cast<clang::NamedDecl*>(decl))};
      return ast->getCanonicalTemplateSpecializationType(templ, args);
    }
  };
} // namespace verona::interop

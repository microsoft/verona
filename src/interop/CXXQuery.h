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
    /// Compiler
    Compiler* Clang;

    /**
     * Simple handler for indirect dispatch on a Clang AST matcher.
     *
     * Use:
     * ```
     *  void myfunc(MatchFinder::MatchResult &);
     *  MatchFinder f;
     *  f.addMatcher(new HandleMatch(myfunc));
     *  f.matchAST(*ast);
     *  // If matches, runs `myfunc` on the matched AST node.
     * ```
     */
    class HandleMatch : public clang::ast_matchers::MatchFinder::MatchCallback
    {
      std::function<void(
        const clang::ast_matchers::MatchFinder::MatchResult& Result)>
        handler;
      void
      run(const clang::ast_matchers::MatchFinder::MatchResult& Result) override
      {
        handler(Result);
      }

    public:
      HandleMatch(
        std::function<
          void(const clang::ast_matchers::MatchFinder::MatchResult& Result)> h)
      : handler(h)
      {}
    };

    /// Maps between CXXType and Clang's types.
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

  public:
    /**
     * CXXQuery c-tor. Creates the internal compile unit, include the
     * user file (and all dependencies), generates the pre-compiled headers,
     * creates the compiler instance and re-attaches the AST to the interface.
     */
    CXXQuery(clang::ASTContext* ast, Compiler* Clang) : ast(ast), Clang(Clang)
    {}

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
      name = "::" + name;
      clang::ast_matchers::MatchFinder finder;
      const clang::EnumDecl* foundEnum = nullptr;
      const clang::CXXRecordDecl* foundClass = nullptr;
      const clang::ClassTemplateDecl* foundTemplateClass = nullptr;

      finder.addMatcher(
        clang::ast_matchers::cxxRecordDecl(clang::ast_matchers::hasName(name))
          .bind("id"),
        new HandleMatch(
          [&](const clang::ast_matchers::MatchFinder::MatchResult& Result) {
            auto* decl = Result.Nodes.getNodeAs<clang::CXXRecordDecl>("id")
                           ->getDefinition();
            if (decl)
            {
              foundClass = decl;
            }
          }));
      finder.addMatcher(
        clang::ast_matchers::classTemplateDecl(
          clang::ast_matchers::hasName(name))
          .bind("id"),
        new HandleMatch(
          [&](const clang::ast_matchers::MatchFinder::MatchResult& Result) {
            auto* decl = Result.Nodes.getNodeAs<clang::ClassTemplateDecl>("id");
            if (decl)
            {
              foundTemplateClass = decl;
            }
          }));
      finder.addMatcher(
        clang::ast_matchers::enumDecl(clang::ast_matchers::hasName(name))
          .bind("id"),
        new HandleMatch(
          [&](const clang::ast_matchers::MatchFinder::MatchResult& Result) {
            auto* decl = Result.Nodes.getNodeAs<clang::EnumDecl>("id");
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

      // If didn't match any type, check for builtins
      return llvm::StringSwitch<CXXType>(name)
        .Case("::bool", CXXType::getBoolean())
        .Case("::unsigned char", CXXType::getUnsignedChar())
        .Case("::char", CXXType::getChar())
        .Case("::signed char", CXXType::getSignedChar())
        .Case("::short", CXXType::getShort())
        .Case("::unsigned short", CXXType::getUnsignedShort())
        .Case("::int", CXXType::getInt())
        .Case("::unsigned int", CXXType::getUnsignedInt())
        .Case("::long", CXXType::getLong())
        .Case("::unsigned long", CXXType::getUnsignedLong())
        .Case("::long long", CXXType::getLongLong())
        .Case("::unsigned long long", CXXType::getUnsignedLongLong())
        .Case("::float", CXXType::getFloat())
        .Case("::double", CXXType::getDouble())
        // Otherwise, just return empty invalid type
        .Default(CXXType());
    }

    /// Return the size in bytes of the specified type.
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

    /// Return the qualified type for a CXXType
    /// FIXME: Do we really need to expose this?
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

    /// Returns the type as a template argument.
    clang::TemplateArgument createTemplateArgumentForType(CXXType t) const
    {
      switch (t.kind)
      {
        case CXXType::Kind::Invalid:
        case CXXType::Kind::TemplateClass:
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

    /// Returns the integral literal as a template value.
    /// TODO: C++20 accepts floating point too
    clang::TemplateArgument createTemplateArgumentForIntegerValue(
      CXXType::BuiltinTypeKinds ty, uint64_t value) const
    {
      assert(CXXType::isIntegral(ty));
      clang::QualType qualTy = typeForBuiltin(ty);
      auto info = ast->getTypeInfo(qualTy);
      llvm::APInt val{static_cast<unsigned int>(info.Width), value};
      auto* literal = clang::IntegerLiteral::Create(
        *ast, val, qualTy, clang::SourceLocation{});
      return clang::TemplateArgument(literal);
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
          templateArgs.push_back(createTemplateArgumentForIntegerValue(
            CXXType::BuiltinTypeKinds::Int, num));
        }
        else
        {
          // Try to find the type name
          auto decl = getType(arg);
          assert(decl.valid());
          templateArgs.push_back(createTemplateArgumentForType(decl));
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
        if (auto typeParam = llvm::dyn_cast<clang::TemplateTypeParmDecl>(param))
        {
          assert(typeParam->hasDefaultArgument());
          templateArgs.push_back(typeParam->getDefaultArgument());
        }
        else if (
          auto nontypeParam =
            llvm::dyn_cast<clang::NonTypeTemplateParmDecl>(param))
        {
          assert(typeParam->hasDefaultArgument());
          templateArgs.push_back(nontypeParam->getDefaultArgument());
        }
      }
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

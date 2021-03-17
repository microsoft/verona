// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "CXXQuery.h"
#include "CXXType.h"
#include "Compiler.h"

namespace verona::interop
{
  /**
   * C++ Builder Interface.
   *
   * This is a class that builds Clang AST expressions. There are a number of
   * helpers to add expressions, calls, types, functions, etc.
   */
  class CXXBuilder
  {
    /// The AST root
    clang::ASTContext* ast = nullptr;
    /// Compiler
    const Compiler* Clang;
    /// Query system
    const CXXQuery* query;

    /**
     * Instantiate the class template specialisation at the end of the main
     * file, if not yet done.
     */
    CXXType instantiateClassTemplate(
      CXXType& classTemplate,
      llvm::ArrayRef<clang::TemplateArgument> args) const
    {
      if (classTemplate.kind != CXXType::Kind::TemplateClass)
      {
        return CXXType{};
      }

      auto& S = Clang->getSema();

      // Check if this specialisation is already present in the AST
      // (declaration, definition, used).
      clang::ClassTemplateDecl* ClassTemplate =
        classTemplate.getAs<clang::ClassTemplateDecl>();
      void* InsertPos = nullptr;
      clang::ClassTemplateSpecializationDecl* Decl =
        ClassTemplate->findSpecialization(args, InsertPos);
      if (!Decl)
      {
        // This is the first time we have referenced this class template
        // specialization. Create the canonical declaration and add it to
        // the set of specializations.
        Decl = clang::ClassTemplateSpecializationDecl::Create(
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
      if (Decl->getSpecializationKind() == clang::TSK_Undeclared)
      {
        clang::MultiLevelTemplateArgumentList TemplateArgLists;
        TemplateArgLists.addOuterTemplateArguments(args);
        S.InstantiateAttrsForDecl(
          TemplateArgLists, ClassTemplate->getTemplatedDecl(), Decl);
      }
      // If specialisation hasn't been defined yet, create its definition at the
      // end of the file.
      clang::ClassTemplateSpecializationDecl* Def =
        clang::cast_or_null<clang::ClassTemplateSpecializationDecl>(
          Decl->getDefinition());
      if (!Def)
      {
        clang::SourceLocation InstantiationLoc = Clang->getEndOfFileLocation();
        assert(InstantiationLoc.isValid());
        S.InstantiateClassTemplateSpecialization(
          InstantiationLoc, Decl, clang::TSK_ExplicitInstantiationDefinition);
        Def = clang::cast<clang::ClassTemplateSpecializationDecl>(
          Decl->getDefinition());
      }
      return CXXType{Def};
    }

    /**
     * Instantiate a new function at the end of the main file, if not yet done.
     */
    clang::FunctionDecl* instantiateFunction(
      const char* name, llvm::ArrayRef<CXXType> args, CXXType ret) const
    {
      auto* DC = ast->getTranslationUnitDecl();
      clang::SourceLocation loc = Clang->getEndOfFileLocation();
      clang::IdentifierInfo& fnNameIdent = ast->Idents.get(name);
      clang::DeclarationName fnName{&fnNameIdent};
      clang::FunctionProtoType::ExtProtoInfo EPI;

      // Get type of args/ret, function
      llvm::SmallVector<clang::QualType> argTys;
      for (auto argTy : args)
        argTys.push_back(query->getQualType(argTy));
      auto retTy = query->getQualType(ret);
      clang::QualType fnTy = ast->getFunctionType(retTy, argTys, EPI);

      // Create a new function
      auto func = clang::FunctionDecl::Create(
        *ast,
        DC,
        loc,
        loc,
        fnName,
        fnTy,
        ast->getTrivialTypeSourceInfo(fnTy),
        clang::StorageClass::SC_None);

      // Define all arguments
      llvm::SmallVector<clang::ParmVarDecl*, 4> argDecls;
      size_t argID = 0;
      for (auto argTy : argTys)
      {
        std::string argName = "arg" + std::to_string(argID++);
        clang::IdentifierInfo& ident = ast->Idents.get(argName);
        clang::ParmVarDecl* arg = clang::ParmVarDecl::Create(
          *ast,
          func,
          loc,
          loc,
          &ident,
          argTy,
          nullptr,
          clang::StorageClass::SC_None,
          nullptr);
        argDecls.push_back(arg);
      }

      // Set function argument list
      func->setParams(argDecls);

      // Associate with the translation unit
      func->setLexicalDeclContext(DC);
      DC->addDecl(func);

      return func;
    }

  public:
    /**
     * CXXInterface c-tor. Creates the internal compile unit, include the
     * user file (and all dependencies), generates the pre-compiled headers,
     * creates the compiler instance and re-attaches the AST to the interface.
     */
    CXXBuilder(clang::ASTContext* ast, Compiler* Clang, const CXXQuery* query)
    : ast(ast), Clang(Clang), query(query)
    {}

    /**
     * Build a template class from a CXXType template and a list of type
     * parameters by name.
     *
     * FIXME: Recursively scan the params for template parameters and define
     * them too.
     */
    CXXType
    buildTemplateType(CXXType& ty, llvm::ArrayRef<std::string> params) const
    {
      // Gather all arguments, passed and default
      auto args = query->gatherTemplateArguments(ty, params);

      // Build the canonical representation
      clang::QualType canon =
        query->getCanonicalTemplateSpecializationType(ty.decl, args);

      // Instantiate and return the definition
      return instantiateClassTemplate(ty, args);
    }

    /**
     * Build a function from name, args and return type, if the function
     * does not yet exist. Return the existing one if it does.
     *
     * FIXME: Should check for template parameters, too.
     */
    clang::FunctionDecl* buildFunction(
      const char* name, llvm::ArrayRef<CXXType> args, CXXType ret) const
    {
      return instantiateFunction(name, args, ret);
    }

    /**
     * Create integer constant literal
     *
     * TODO: Create all the ones we use in the template specialisation
     */
    clang::IntegerLiteral*
    createIntegerLiteral(unsigned int len, unsigned long val) const
    {
      llvm::APInt num{len, val};
      auto* lit = clang::IntegerLiteral::Create(
        *ast,
        num,
        query->getQualType(CXXType::getInt()),
        clang::SourceLocation{});
      return lit;
    }

    /**
     * Create a return instruction
     *
     * TODO: Create all the ones we use in code generation
     */
    clang::ReturnStmt*
    createReturn(clang::Expr* val, clang::FunctionDecl* func) const
    {
      auto retStmt =
        clang::ReturnStmt::Create(*ast, func->getLocation(), val, nullptr);
      func->setBody(retStmt);
      return retStmt;
    }
  };
} // namespace verona::interop

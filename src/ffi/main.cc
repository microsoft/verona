// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "CXXInterface.h"

using namespace verona::ffi;

/// Looks up a symbol from a CXX interface by name
void test(CXXInterface& interface, std::string& name)
{
  auto t = TimeReport(std::string("Lookup time"));
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
    fprintf(stderr, "Not found: %s\n", name.c_str());
  }

  // For template types, try to find their default arguments
  if (decl.kind == CXXType::Kind::TemplateClass)
  {
    auto irb = interface.getType(name);
    auto params =
      dyn_cast<ClassTemplateDecl>(irb.decl)->getTemplateParameters();
    std::vector<TemplateArgument> args;
    for (auto param : *params)
    {
      if (auto typeParam = dyn_cast<TemplateTypeParmDecl>(param))
      {
        if (typeParam->hasDefaultArgument())
          args.push_back(typeParam->getDefaultArgument());
      }
      else if (auto nontypeParam = dyn_cast<NonTypeTemplateParmDecl>(param))
      {
        if (nontypeParam->hasDefaultArgument())
          args.push_back(nontypeParam->getDefaultArgument());
      }
    }
    if (args.size())
    {
      TemplateName irbName{
        dyn_cast<TemplateDecl>(const_cast<NamedDecl*>(irb.decl))};

      interface.getAST()->getTemplateSpecializationType(irbName, args).dump();
    }
  }
}

/// Simple syntax error
void syntax_and_die()
{
  puts("Syntax: verona-ffi <cxx-filename> <symbol1> <symbol2> ... <symbolN>");
  exit(1);
}

int main(int argc, char** argv)
{
  // For now, the command line arguments are simple, so we keep it stupid.
  std::string file;
  if (argc > 1 && argv[1] != 0)
  {
    file = argv[1];
  }
  std::vector<std::string> symbols;
  if (argc > 2)
  {
    for (size_t i = 2; i < argc; i++)
      symbols.push_back(argv[i]);
  }
  if (file.empty() || symbols.empty())
  {
    syntax_and_die();
  }

  // FIXME: Verona compiler should be able to find the path and pass include
  // paths to this interface.
  CXXInterface interface(file);

  // FIXME: We should be able to pass a list and get a list back.
  for (auto symbol : symbols)
    test(interface, symbol);

  exit(0);

  // The remaining of this file will be slowly moved up when functionality is
  // available for each one of them. Most of it was assuming the file in the
  // interface was IRBuilder.h in the LLVM repo.
  /*
  CXXInterface stdarray("/usr/include/c++/v1/array");
  auto t = TimeReport("Instantiating std::array");
  auto arr = stdarray.getType("std::array");
  fprintf(
    stderr,
    "std::array has %d template parameters\n",
    arr.numberOfTemplateParameters());
  auto IntTy = CXXType{CXXType::BuiltinTypeKinds::Int};
  auto TypeArg = stdarray.createTemplateArgumentForType(IntTy);
  auto ValueArg = stdarray.createTemplateArgumentForIntegerValue(
    CXXType::BuiltinTypeKinds::Int, 4);

  CXXType arrIntFour =
    stdarray.instantiateClassTemplate(arr, {TypeArg, ValueArg});
  arr.decl->dump();
  arrIntFour.decl->dump();
  fprintf(
    stderr,
    "std::array<int, 4> is %lu bytes\n",
    stdarray.getTypeSize(arrIntFour));

  auto* DC = stdarray.ast->getTranslationUnitDecl();
  auto& SM = stdarray.Clang->getSourceManager();
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
  */
  return 0;
}

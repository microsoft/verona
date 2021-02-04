// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "CXXInterface.h"

using namespace verona::ffi;

/// Looks up a symbol from a CXX interface by name
/// Tested on <array> looking for type "std::array"
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
    return;
  }

  // For template types, try to find their parameters and default arguments
  if (decl.kind == CXXType::Kind::TemplateClass)
  {
    fprintf(
      stderr,
      "%s has %d template parameters\n",
      name.c_str(),
      decl.numberOfTemplateParameters());

    // Try to fit `int` to the parameter
    auto IntTy = CXXType{CXXType::BuiltinTypeKinds::Int};
    auto TypeArg = interface.createTemplateArgumentForType(IntTy);
    auto ValueArg = interface.createTemplateArgumentForIntegerValue(
      CXXType::BuiltinTypeKinds::Int, 4);

    // Find any default parameter
    auto params =
      dyn_cast<ClassTemplateDecl>(decl.decl)->getTemplateParameters();
    std::vector<TemplateArgument> args;
    std::vector<TemplateArgument> defaultArgs;
    for (auto param : *params)
    {
      if (auto typeParam = dyn_cast<TemplateTypeParmDecl>(param))
      {
        args.push_back(TypeArg);
        if (typeParam->hasDefaultArgument())
          defaultArgs.push_back(typeParam->getDefaultArgument());
      }
      else if (auto nontypeParam = dyn_cast<NonTypeTemplateParmDecl>(param))
      {
        args.push_back(ValueArg);
        if (nontypeParam->hasDefaultArgument())
          defaultArgs.push_back(nontypeParam->getDefaultArgument());
      }
    }
    if (defaultArgs.size())
    {
      // Shows the default arguments, if any
      TemplateName irbName{
        dyn_cast<TemplateDecl>(const_cast<NamedDecl*>(decl.decl))};
      interface.getAST()
        ->getTemplateSpecializationType(irbName, defaultArgs)
        .dump();
    }
    if (args.size())
    {
      // Tries to instantiate a full specialisation
      // NOTE: This only works for integer type/arguments
      CXXType spec = interface.instantiateClassTemplate(decl, args);
      fprintf(
        stderr,
        "%s<int> is %lu bytes\n",
        name.c_str(),
        interface.getTypeSize(spec));
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
  SourceLocation loc = interface.getCompiler()->getEndOfFileLocation();
  IdentifierInfo& fnNameIdent =
    interface.getAST()->Idents.get("verona_wrapper_fn_1");
  DeclarationName fnName{&fnNameIdent};
  QualType retTy = interface.getAST()->IntTy;
  QualType argTy = interface.getAST()->IntTy;
  FunctionProtoType::ExtProtoInfo EPI;
  QualType fnTy = interface.getAST()->getFunctionType(retTy, {argTy}, EPI);

  auto newFunction = FunctionDecl::Create(
    *interface.getAST(),
    DC,
    loc,
    loc,
    fnName,
    fnTy,
    interface.getAST()->getTrivialTypeSourceInfo(fnTy),
    StorageClass::SC_None);
  newFunction->setLexicalDeclContext(DC);

  IdentifierInfo& arg1Ident = interface.getAST()->Idents.get("arg1");
  ParmVarDecl* arg1Decl = ParmVarDecl::Create(
    *interface.getAST(),
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
    *interface.getAST(), four, interface.getAST()->IntTy, SourceLocation{});
  auto retStmt = ReturnStmt::Create(*interface.getAST(), loc, fourLiteral,
  nullptr); newFunction->setBody(retStmt);

  newFunction->dump();
  DC->addDecl(newFunction);

  // DC->dump();

  // Decl->dump();
  // Def->dump();

  interface.emitLLVM();

  TemplateName arrName{arr.getAs<TemplateDecl>()};
  interface.getAST()
    ->getCanonicalTemplateSpecializationType(arrName, {TypeArg, ValueArg})
    .dump();
  interface.getAST()->
    ->getCanonicalTemplateSpecializationType(arrName, {TypeArg, TypeArg})
    .dump();
  */
  return 0;
}

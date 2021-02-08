// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "CXXInterface.h"

using namespace verona::ffi;

void printType(CXXType& ty)
{
  assert(ty.kind != CXXType::Kind::Invalid);
  auto* d = ty.decl;
  auto kind = ty.kindName();
  auto name = d->getName().str();
  fprintf(stderr, "%s (@%p) %s", name.c_str(), d, kind);
  if (ty.kind == CXXType::Kind::Builtin)
    fprintf(stderr, "(%s)", ty.builtinKindName());
  fprintf(stderr, "\n");
}

/// Looks up a symbol from a CXX interface by name
/// Tested on <array> looking for type "std::array"
void test_type(CXXInterface& interface, std::string& name)
{
  auto t = TimeReport(std::string("Lookup time"));
  auto decl = interface.getType(name);
  auto* d = decl.decl;
  if (decl.kind != CXXType::Kind::Invalid)
  {
    fprintf(stderr, "Found: ");
    printType(decl);
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
    auto TypeArg = interface.createTemplateArgumentForType(CXXType::getInt());
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
    // If all args have a default, instantiate the default implementation
    if (defaultArgs.size() && defaultArgs.size() == args.size())
    {
      // Shows the default arguments, if any
      QualType spec =
        interface.getTemplateSpecializationType(decl.decl, defaultArgs);
      spec.dump();
    }
    // Tries to instantiate a full specialisation
    // NOTE: This only works for integer type/arguments
    if (args.size())
    {
      CXXType spec = interface.instantiateClassTemplate(decl, args);
      fprintf(
        stderr,
        "%s<int> is %lu bytes\n",
        name.c_str(),
        interface.getTypeSize(spec));
    }
  }
}

clang::FunctionDecl* test_function(CXXInterface& interface)
{
  // Create a new function on the main file
  auto intTy = CXXType::getInt();
  llvm::SmallVector<CXXType, 1> args{intTy};

  // Create new function
  auto func = interface.instantiateFunction("verona_wrapper_fn_1", args, intTy);

  // Set first argument
  auto arg = interface.createFunctionArgument("arg1", intTy, func);

  // Create constant literal
  auto* fourLiteral = interface.createIntegerLiteral(32, 4);

  // Return statement
  interface.createReturn(fourLiteral, func);

  func->dump();
  return func;
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
  if (file.empty())
  {
    puts("Syntax: verona-ffi <cxx-filename> <symbol1> <symbol2> ... <symbolN>");
    exit(1);
  }

  // FIXME: Verona compiler should be able to find the path and pass include
  // paths to this interface.
  CXXInterface interface(file);

  // FIXME: We should be able to pass a list and get a list back.
  if (symbols.size())
  {
    fprintf(stderr, "\nQuerying some types...\n");
    for (auto symbol : symbols)
      test_type(interface, symbol);
  }

  // Test function creation
  fprintf(stderr, "\nCreating a new function...\n");
  test_function(interface);

  // Emit whatever is left on the main file
  fprintf(stderr, "\nGenerating LLVM IR...\n");
  auto mod = interface.emitLLVM();
  mod->dump();

  // The remaining of this file will be slowly moved up when functionality is
  // available for each one of them. Most of it was assuming the file in the
  // interface was IRBuilder.h in the LLVM repo.
  /*
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

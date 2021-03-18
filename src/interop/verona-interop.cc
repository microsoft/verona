// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "CXXInterface.h"

#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;
using namespace verona::interop;
using namespace clang;
namespace cl = llvm::cl;

/**
 * This file is a helper for a few tests, not the aim of an actual
 * interoperability driver, which will actually be hidden inside the compiler.
 *
 * We should move this into a bunch of unit tests andn run them directly from
 * ctest, with all the functionality we'll need from the sandbox code inside
 * the compiler.
 */

namespace
{
  // For help's sake, will never be parsed, as we intercept
  cl::opt<string> config(
    "config",
    cl::desc("<config file>"),
    cl::Optional,
    cl::value_desc("config"));

  // Test function (TODO: make this more generic)
  cl::opt<bool> testFunction(
    "function",
    cl::desc("Creates a test function"),
    cl::Optional,
    cl::init(false));

  cl::opt<bool> dumpIR(
    "dump",
    cl::desc("Dumps the whole IR at the end"),
    cl::Optional,
    cl::init(false));

  cl::opt<string> inputFile(
    cl::Positional,
    cl::desc("<input file>"),
    cl::init("-"),
    cl::value_desc("filename"));

  cl::opt<string> symbol(
    cl::Positional,
    cl::desc("<symbol>"),
    cl::init(""),
    cl::value_desc("symbol"));

  cl::list<string> specialization(
    "params",
    cl::desc("<template specialization parameters>"),
    cl::CommaSeparated,
    cl::value_desc("specialization"));

  cl::list<string> fields(
    "fields",
    cl::desc("<list of fields to query>"),
    cl::CommaSeparated,
    cl::value_desc("fields"));

  cl::opt<string> method(
    "method",
    cl::desc("<single method to query>"),
    cl::Optional,
    cl::value_desc("method"));

  cl::list<string> argTys(
    "argTys",
    cl::desc("<list of method's argument types to query>"),
    cl::CommaSeparated,
    cl::value_desc("argTys"));

  /// Add new option to arguments array
  void addArgOption(vector<char*>& args, char* arg, size_t len)
  {
    args.push_back(new char[len + 1]);
    auto& inplace = args[args.size() - 1];
    copy(arg, arg + len, inplace);
    inplace[len] = 0;
  }

  /// Parse config file adding args to the args globals
  void parseCommandLine(int argc, char** argv, vector<string>& includePath)
  {
    // Replace "--config file" with the contents of file
    vector<char*> args;
    string configFileName;
    StringRef flag("-config");
    for (int i = 0; i < argc; i++)
    {
      auto arg = argv[i];
      // If not config, just copy the argv as is
      if (!flag.equals(arg))
      {
        addArgOption(args, arg, strlen(arg));
        continue;
      }

      // Else, append all arguments from the file to args
      string configFile(argv[++i]);
      ifstream file(configFile);
      if (!file.good())
      {
        cerr << "Error opening config file " << configFile.c_str() << endl;
        exit(1);
      }

      // For each arg, append the options to the command line
      string buffer;
      while (file >> quoted(buffer))
      {
        addArgOption(args, buffer.data(), buffer.size());
      }
      file.close();

      // Add the path to the config file to the include path
      auto conf = FileSystem::getRealPath(configFile);
      auto dir = FileSystem::getDirName(conf);
      includePath.push_back(dir);
    }

    // Parse the command line
    cl::ParseCommandLineOptions(
      args.size(), args.data(), "Verona Interop test\n");
  }

  /// Test call
  void test_call(
    CXXType& context,
    clang::CXXMethodDecl* func,
    llvm::ArrayRef<clang::QualType> argTys,
    clang::QualType retTy,
    const CXXInterface& interface)
  {
    const CXXQuery* query = interface.getQuery();
    const CXXBuilder* builder = interface.getBuilder();

    // Build a unique name (per class/method)
    string fqName = context.getName().str();
    if (context.isTemplate())
    {
      auto params = context.getTemplateParameters();
      if (params)
      {
        for (auto* decl : *params)
        {
          fqName += "_" + decl->getNameAsString();
        }
      }
      fqName += "_";
    }
    fqName += func->getName().str();
    string wrapperName = "__call_to_" + fqName;

    // Create a function with a hygienic name and the same args
    auto caller = builder->buildFunction(wrapperName, argTys, retTy);

    // Collect arguments
    auto args = caller->parameters();

    // Create the call to the actual function
    auto call = builder->createMemberCall(func, args, retTy, caller);

    // Return the call's value
    builder->createReturn(call, caller);
  }

  /// Test a type
  void test_type(
    llvm::StringRef name,
    llvm::ArrayRef<std::string> args,
    llvm::ArrayRef<std::string> fields,
    const CXXInterface& interface)
  {
    const CXXQuery* query = interface.getQuery();
    const CXXBuilder* builder = interface.getBuilder();

    // Find type
    CXXType ty = query->getType(symbol);
    if (!ty.valid())
    {
      cerr << "Invalid type '" << ty.getName().str() << "'" << endl;
      exit(1);
    }

    // Print type name and kind
    cout << "Type '" << ty.getName().str() << "' as " << ty.kindName();
    if (ty.kind == CXXType::Kind::Builtin)
      cout << " (" << ty.builtinKindName() << ")";
    cout << endl;

    // Try and specialize a template
    // TODO: Should this be part of getType()?
    // Do we need a complete type for template parameters?
    if (ty.isTemplate())
    {
      // Tries to instantiate a full specialisation
      ty = builder->buildTemplateType(ty, specialization);
    }

    // If all goes well, this returns a platform-dependent size
    cout << "Size of " << ty.getName().str() << " is " << query->getTypeSize(ty)
         << " bytes" << endl;

    // If requested any field to lookup, by name
    for (auto f : fields)
    {
      auto field = query->getField(ty, f);
      if (!field)
      {
        cerr << "Invalid field '" << f << "' on type '" << ty.getName().str()
             << "'" << endl;
        exit(1);
      }
      auto fieldTy = field->getType();
      auto tyClass = fieldTy->getTypeClassName();
      auto tyName = fieldTy.getAsString();
      cout << "Field '" << field->getName().str() << "' has " << tyClass
           << " type '" << tyName << "'" << endl;
    }

    // If requested any method to lookup, by name and arg types
    if (!method.empty())
    {
      auto func = query->getMethod(ty, method, argTys);
      if (!func)
      {
        cerr << "Invalid method '" << method << "' on type '"
             << ty.getName().str() << "'" << endl;
        exit(1);
      }
      auto fName = func->getName().str();
      cout << "Method '" << fName << "' with signature: (";
      llvm::SmallVector<clang::QualType, 1> argTys;
      for (auto arg : func->parameters())
      {
        if (arg != *func->param_begin())
          cout << ", ";
        auto argTy = arg->getType();
        argTys.push_back(argTy);
        cout << argTy.getAsString() << " " << arg->getName().str();
      }
      auto retTy = func->getReturnType();
      cout << ") -> " << retTy.getAsString() << endl;

      // Instantiate function in AST that calls this method
      test_call(ty, func, argTys, retTy, interface);
    }
  }

  /// Creates a test function
  void test_function(const char* name, const CXXInterface& interface)
  {
    const CXXQuery* query = interface.getQuery();
    const CXXBuilder* builder = interface.getBuilder();

    // Create a new function on the main file
    auto intTy = query->getQualType(CXXType::getInt());
    llvm::SmallVector<clang::QualType, 1> args{intTy};

    // Create new function
    auto func = builder->buildFunction(name, args, intTy);

    // Create constant literal
    auto* fourLiteral = builder->createIntegerLiteral(32, 4);

    // Return statement
    builder->createReturn(fourLiteral, func);
  }
} // namespace

int main(int argc, char** argv)
{
  // Parse cmd-line options
  vector<string> includePath;
  parseCommandLine(argc, argv, includePath);

  // Create the C++ interface
  CXXInterface interface(inputFile, includePath);

  // Test type query
  if (!symbol.empty())
  {
    test_type(symbol, specialization, fields, interface);
  }

  // Test function creation
  if (testFunction)
  {
    test_function("verona_wrapper_fn_1", interface);
  }

  // Dumps the AST before trying to emit LLVM for debugging purposes
  // NOTE: Output is not stable, don't use it for tests
  if (dumpIR)
  {
    interface.dumpAST();
  }

  // Emit whatever is left on the main file
  // This is silent, just to make sure nothing breaks here
  auto mod = interface.emitLLVM();

  // Dump LLVM IR for debugging purposes
  // NOTE: Output is not stable, don't use it for tests
  if (dumpIR)
  {
    mod->dump();
  }

  return 0;
}

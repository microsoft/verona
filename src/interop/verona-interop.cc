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
    cl::Positional,
    cl::desc("<template specialization parameters>"),
    cl::CommaSeparated,
    cl::value_desc("specialization"));

  /// Prints a type to stdout
  void printType(CXXType& ty)
  {
    assert(ty.valid());
    auto kind = ty.kindName();
    auto name = ty.getName().str();
    cout << name << " " << kind;
    if (ty.kind == CXXType::Kind::Builtin)
      cout << "(" << ty.builtinKindName() << ")";
    cout << endl;
  }

  /// Looks up a symbol from a CXX interface by name
  /// Tested on <array> looking for type "array"
  CXXType get_type(const CXXQuery* query, string& name)
  {
    auto ty = query->getType(name);
    if (ty.valid())
    {
      cout << "Found: ";
      printType(ty);
    }
    else
    {
      cout << "Not found: " << name.c_str() << endl;
    }
    return ty;
  }

  /// Creates a test function
  clang::FunctionDecl*
  test_function(const CXXBuilder* builder, const char* name)
  {
    // Create a new function on the main file
    auto intTy = CXXType::getInt();
    llvm::SmallVector<CXXType, 1> args{intTy};

    // Create new function
    auto func = builder->instantiateFunction(name, args, intTy);

    // Create constant literal
    auto* fourLiteral = builder->createIntegerLiteral(32, 4);

    // Return statement
    builder->createReturn(fourLiteral, func);

    return func;
  }

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
    StringRef flag("--config");
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
} // namespace

int main(int argc, char** argv)
{
  // Parse cmd-line options
  vector<string> includePath;
  parseCommandLine(argc, argv, includePath);

  // Create the C++ interface
  CXXInterface interface(inputFile, includePath);
  const CXXQuery* query = interface.getQuery();
  const CXXBuilder* builder = interface.getBuilder();

  // Test type query
  if (!symbol.empty())
  {
    // Query the requested symbol
    auto ty = get_type(query, symbol);

    // Try and specialize a template
    uint64_t req = specialization.size();
    if (req || ty.isTemplate())
    {
      // Make sure this is a template class
      if (!ty.isTemplate())
      {
        cerr << "Class " << symbol.c_str()
             << " is not a template class, can't specialize" << endl;
        exit(1);
      }

      // Specialize the template with the arguments
      auto args = query->gatherTemplateArguments(ty, specialization);
      // Canonical representation
      QualType canon =
        query->getCanonicalTemplateSpecializationType(ty.decl, args);

      // Tries to instantiate a full specialisation
      auto spec = builder->instantiateClassTemplate(ty, args);

      cout << "Size of " << spec.getName().str() << " is "
           << query->getTypeSize(spec) << " bytes" << endl;
    }
  }

  // Test function creation
  if (testFunction)
  {
    test_function(builder, "verona_wrapper_fn_1");
  }

  // Emit whatever is left on the main file
  // This is silent, just to make sure nothing breaks here
  auto mod = interface.emitLLVM();

  // This just checks that the function was generated
  if (testFunction)
  {
    assert(mod->getFunction("_Z19verona_wrapper_fn_1i"));
  }

  // This just dumps everything, for debugging purposes
  if (dumpIR)
  {
    mod->dump();
  }

  return 0;
}

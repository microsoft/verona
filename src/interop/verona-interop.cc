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
    cl::desc("<list of filed to query>"),
    cl::CommaSeparated,
    cl::value_desc("fields"));

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

  /// Test a type
  void test_type(
    llvm::StringRef name,
    llvm::ArrayRef<std::string> args,
    llvm::ArrayRef<std::string> fields,
    const CXXQuery* query,
    const CXXBuilder* builder)
  {
    // Find type
    CXXType ty = query->getType(symbol);
    if (!ty.valid())
    {
      cerr << "Invalid type '" << ty.getName().str() << "'" << endl;
      exit(1);
    }

    // Print type name and kind
    cout << "Found: ";
    printType(ty);

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
  }

  /// Creates a test function
  void test_function(const char* name, const CXXBuilder* builder)
  {
    // Create a new function on the main file
    auto intTy = CXXType::getInt();
    llvm::SmallVector<CXXType, 1> args{intTy};

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
  const CXXQuery* query = interface.getQuery();
  const CXXBuilder* builder = interface.getBuilder();

  // Test type query
  if (!symbol.empty())
  {
    test_type(symbol, specialization, fields, query, builder);
  }

  // Test function creation
  if (testFunction)
  {
    test_function("verona_wrapper_fn_1", builder);
  }

  // Emit whatever is left on the main file
  // This is silent, just to make sure nothing breaks here
  auto mod = interface.emitLLVM();

  // This just dumps everything, for debugging purposes
  // NOTE: Output is not stable, don't use it for tests
  if (dumpIR)
  {
    mod->dump();
  }

  return 0;
}

// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "CXXInterface.h"

#include <fstream>
#include <iomanip>

using namespace std;
using namespace verona::ffi;
using namespace clang;
namespace cl = llvm::cl;

namespace
{
  // Test function (TODO: make this more generic)
  cl::opt<bool> testFunction(
    "function",
    cl::desc("Creates a test function"),
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
    printf("%s (@%p) %s", name.c_str(), ty.decl, kind);
    if (ty.kind == CXXType::Kind::Builtin)
      printf("(%s)", ty.builtinKindName());
    printf("\n");
  }

  /// Looks up a symbol from a CXX interface by name
  /// Tested on <array> looking for type "array"
  CXXType get_type(CXXInterface& interface, string& name)
  {
    auto ty = interface.getType(name);
    if (ty.valid())
    {
      printf("Found: ");
      printType(ty);
    }
    else
    {
      printf("Not found: %s\n", name.c_str());
    }
    return ty;
  }

  /// Create the parameters of a template class from type names or values
  vector<TemplateArgument>
  create_template_args(CXXInterface& interface, llvm::ArrayRef<string> args)
  {
    vector<TemplateArgument> templateArgs;
    for (auto arg : args)
    {
      if (isdigit(arg[0]))
      {
        // Numbers default to int parameter
        auto num = atol(arg.c_str());
        templateArgs.push_back(interface.createTemplateArgumentForIntegerValue(
          CXXType::BuiltinTypeKinds::Int, num));
      }
      else
      {
        // Try to find the type name
        auto decl = interface.getType(arg);
        if (!decl.valid())
        {
          fprintf(
            stderr, "Invalid template specialization type %s\n", arg.c_str());
          exit(1);
        }
        templateArgs.push_back(interface.createTemplateArgumentForType(decl));
      }
    }
    return templateArgs;
  }

  /// Specialize the template into a CXXType
  CXXType specialize_template(
    CXXInterface& interface, CXXType& ty, llvm::ArrayRef<TemplateArgument> args)
  {
    // Canonical representation
    printf("Canonical Template specialisation:\n");
    QualType canon =
      interface.getCanonicalTemplateSpecializationType(ty.decl, args);
    canon.dump();

    // Tries to instantiate a full specialisation
    return interface.instantiateClassTemplate(ty, args);
  }

  /// Creates a test function (TODO: make this more generic)
  clang::FunctionDecl* test_function(CXXInterface& interface)
  {
    // Create a new function on the main file
    auto intTy = CXXType::getInt();
    llvm::SmallVector<CXXType, 1> args{intTy};

    printf("Simple function:\n");
    // Create new function
    auto func =
      interface.instantiateFunction("verona_wrapper_fn_1", args, intTy);

    // Set first argument
    auto arg = interface.createFunctionArgument("arg1", intTy, func);

    // Create constant literal
    auto* fourLiteral = interface.createIntegerLiteral(32, 4);

    // Return statement
    interface.createReturn(fourLiteral, func);

    func->dump();
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
        fprintf(stderr, "Error opening config file %s\n", configFile.c_str());
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
    cl::ParseCommandLineOptions(args.size(), args.data(), "Verona FFI test\n");
  }
} // namespace

int main(int argc, char** argv)
{
  // Parse cmd-line options
  vector<string> includePath;
  parseCommandLine(argc, argv, includePath);

  // Create the C++ interface
  CXXInterface interface(inputFile, includePath);

  // Test function creation
  // TODO: Make this more parametric
  if (testFunction)
  {
    test_function(interface);
  }

  if (!symbol.empty())
  {
    // Query the requested symbol
    auto decl = get_type(interface, symbol);

    // Try and specialize a template
    auto req = specialization.size();
    if (req)
    {
      // Make sure this is a template class
      if (!decl.isTemplate())
      {
        fprintf(
          stderr,
          "Class %s is not a template class, can't specialize",
          symbol.c_str());
        exit(1);
      }

      // Make sure the number of arguments is the same
      auto has = decl.numberOfTemplateParameters();
      if (req != has)
      {
        fprintf(
          stderr,
          "Requested %ld template arguments but class %s only has "
          "%d\n",
          req,
          symbol.c_str(),
          has);
        exit(1);
      }

      // Specialize the template with the arguments
      auto args = create_template_args(interface, specialization);
      auto spec = specialize_template(interface, decl, args);
      printf(
        "Size of %s is %lu bytes\n",
        spec.getName().str().c_str(),
        interface.getTypeSize(spec));
    }
  }

  // Emit whatever is left on the main file
  auto mod = interface.emitLLVM();
  mod->dump();

  return 0;
}

// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "config.h"
#include "driver.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "parser/anf.h"
#include "parser/dnf.h"
#include "parser/parser.h"
#include "parser/resolve.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"

#include <iostream>

using namespace std;
namespace cl = llvm::cl;

namespace
{
  /// For help's sake, will never be parsed, as we intercept
  cl::opt<string> config(
    "config",
    cl::desc("<config file>"),
    cl::Optional,
    cl::value_desc("config"));

  /// Input file name (- means stdin)
  cl::opt<std::string> inputFile(
    cl::Positional,
    cl::desc("<input file>"),
    cl::init("-"),
    cl::value_desc("filename"));

  /// Source file kind
  enum class InputKind
  {
    None,
    Verona,
    MLIR
  };
  /// Source file kind option
  cl::opt<enum InputKind> inputKind(
    "x",
    cl::init(InputKind::None),
    cl::desc("Input type"),
    cl::values(clEnumValN(InputKind::Verona, "verona", "Verona file")),
    cl::values(clEnumValN(InputKind::MLIR, "mlir", "MLIR file")));

  /// Optimisations enabled
  cl::opt<unsigned> optLevel(
    "O",
    cl::desc("Optimization level. [-O0, -O1, -O2, or -O3] "
             "(default = '-O0')"),
    cl::Prefix,
    cl::ZeroOrMore,
    cl::init(0));

  /// Which output to emit
  cl::opt<std::string> outputFmt(
    "out",
    cl::desc("Output format [mlir, ll, jit] "
             "(default = 'mlir')"),
    cl::Prefix,
    cl::ZeroOrMore,
    cl::init("mlir"));

  /// Test only, redirect output to /dev/null
  cl::opt<bool> testOnly(
    "t", cl::desc("Test only (no output)"), cl::Optional, cl::init(false));

  /// Output file name (- means stdout)
  cl::opt<std::string> outputFile("o", cl::init(""), cl::desc("Output file"));

  /// Appends .new to extension if the new extension is the same
  void addExtension(llvm::SmallString<128>& name, llvm::StringRef ext)
  {
    auto currExt = llvm::sys::path::extension(name);
    if (ext.equals(currExt))
      llvm::sys::path::replace_extension(name, ".new");
    else
      llvm::sys::path::replace_extension(name, ext);
  }

  /// Set defaults form command line arguments
  void cmdLineDefaults(const char* execName)
  {
    // Detect source type from extension, if not passed as argument
    if (inputKind == InputKind::None)
    {
      llvm::StringRef filename(inputFile);
      if (filename.endswith(".verona"))
        inputKind = InputKind::Verona;
      else if (filename.endswith(".mlir"))
        inputKind = InputKind::MLIR;
      else if (filename == "-") // STDIN, assume Verona
        inputKind = InputKind::Verona;
    }

    // Choose output file extension from output type
    if (outputFile.empty())
    {
      // Default input file is "-"
      if (testOnly)
      {
        // No output
        outputFile = "";
      }
      else if (inputFile == "-")
      {
        // stdin defaults to stdout
        outputFile = "-";
      }
      else
      {
        // Extension derives from output kind
        llvm::SmallString<128> newName(inputFile);
        addExtension(newName, outputFmt);
        outputFile = newName.c_str();
      }
    }
  }

  /// Get Verona std library path
  std::string getStdLibPath(const char* progName)
  {
    std::string exec = llvm::sys::fs::getMainExecutable(
      progName, /*some function in this binary*/ (void*)getStdLibPath);
    std::string path =
      std::string(llvm::sys::path::parent_path(exec)) + "/stdlib/";
    return path;
  }

  /// Parse config file adding args to the args globals
  void parseCommandLine(int argc, char** argv)
  {
    // Replace "--config file" with the contents of file
    CmdLineAppend app;
    if (!app.parse(argc, argv))
    {
      auto paths = app.configPaths();
      // Whatever error was on the last config file
      auto lastConfig = paths[paths.size() - 1];
      cerr << "Error opening config file " << lastConfig.c_str() << endl;
      exit(1);
    }

    // Parse the command line
    cl::ParseCommandLineOptions(
      app.argc(), app.argv(), "Verona MLIR Generator\n");
  }
} // namespace

using namespace verona::parser;
using namespace mlir::verona;

int main(int argc, char** argv)
{
  // Set up pretty-print signal handlers
  llvm::InitLLVM y(argc, argv);

  // Parse cmd-line options
  parseCommandLine(argc, argv);
  cmdLineDefaults(argv[0]);

  if (inputKind == InputKind::None)
  {
    std::cerr << "ERROR: Unknown source type for '" << inputFile
              << "'. Must be [verona, mlir]" << std::endl;
    return 1;
  }

  mlir::verona::Driver driver(optLevel);
  llvm::ExitOnError check;

  // Parse the source file (verona/mlir)
  mlir::OwningModuleRef module;
  mlir::MLIRContext context;
  switch (inputKind)
  {
    case InputKind::Verona:
    {
      // Parse the file
      auto stdlibpath = getStdLibPath(argv[0]);
      auto [ok, ast] = parse(inputFile, stdlibpath);
      ok = ok && dnf::wellformed(ast);

      // Resolve types
      ok = ok && resolve::run(ast);
      ok = ok && resolve::wellformed(ast);

      // Convert to A-norm
      ok = ok && anf::run(ast);
      ok = ok && anf::wellformed(ast);

      if (!ok)
      {
        std::cerr << "ERROR: cannot parse Verona file " << inputFile
                  << std::endl;
        return 1;
      }

      // Parse AST file into MLIR
      check(driver.readAST(ast));
    }
    break;
    case InputKind::MLIR:
      // Parse MLIR file
      check(driver.readMLIR(inputFile));
      break;
    default:
      std::cerr << "ERROR: invalid source file type" << std::endl;
      return 1;
  }

  // Dumps the module in the chosen format
  if (outputFmt == "mlir")
  {
    check(driver.emitMLIR(outputFile));
  }
  else if (outputFmt == "ll")
  {
    check(driver.emitLLVM(outputFile));
  }
  else if (outputFmt == "jit")
  {
    int returnValue = 0;
    check(driver.runLLVM(returnValue));
    std::cout << "Return value: " << returnValue << std::endl;
  }
  else
  {
    std::cerr << "Output format " << outputFmt << " not yet supported"
              << std::endl;
    return 1;
  }

  return 0;
}

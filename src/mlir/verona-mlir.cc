// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "driver.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "parser/anf.h"
#include "parser/dnf.h"
#include "parser/parser.h"
#include "parser/resolve.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"

#include <iostream>

namespace
{
  namespace cl = llvm::cl;
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
  static cl::opt<unsigned> optLevel(
    "O",
    cl::desc("Optimization level. [-O0, -O1, -O2, or -O3] "
             "(default = '-O0')"),
    cl::Prefix,
    cl::ZeroOrMore,
    cl::init(0));

  /// Which output to emit
  static cl::opt<std::string> outputFmt(
    "out",
    cl::desc("Output format [mlir, llvm, asm, obj] "
             "(default = 'mlir')"),
    cl::Prefix,
    cl::ZeroOrMore,
    cl::init("mlir"));

  /// Output file name (- means stdout)
  cl::opt<std::string> outputFile("o", cl::init(""), cl::desc("Output file"));

  /// Set defaults form command line arguments
  void cmdLineDefaults(const char* execName)
  {
    // Default input is stdin
    if (inputFile.empty())
      inputFile = "-";

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
    // Careful with mlir->mlir not to overwrite source file
    if (outputFile.empty())
    {
      if (inputFile == "-")
      {
        outputFile = "-";
      }
      else
      {
        llvm::SmallString<128> newName(inputFile);
        if (inputKind == InputKind::MLIR)
          llvm::sys::path::replace_extension(newName, ".new.mlir");
        else
          llvm::sys::path::replace_extension(newName, ".mlir");
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
} // namespace

using namespace verona::parser;
using namespace mlir::verona;

int main(int argc, char** argv)
{
  // Set up pretty-print signal handlers
  llvm::InitLLVM y(argc, argv);

  // Parse cmd-line options
  cl::ParseCommandLineOptions(argc, argv, "Verona MLIR Generator\n");
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
  else if (outputFmt == "llvm")
  {
    check(driver.emitLLVM(outputFile));
  }
  else
  {
    std::cerr << "Output format " << outputFmt << " not yet supported"
              << std::endl;
    return 1;
  }

  return 0;
}

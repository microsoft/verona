// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "CLI/CLI.hpp"
#include "ast/module.h"
#include "ast/parser.h"
#include "ast/path.h"
#include "ast/prec.h"
#include "ast/ref.h"
#include "ast/sym.h"
#include "dialect/VeronaDialect.h"
#include "generator.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

namespace
{
  namespace cl = llvm::cl;
  // Input file name
  cl::opt<std::string> inputFile(
    cl::Positional,
    cl::desc("<input file>"),
    cl::init("-"),
    cl::value_desc("filename"));

  // Source file types, to choose how to parse
  enum class InputKind
  {
    None,
    Verona,
    MLIR
  };
  cl::opt<enum InputKind> inputKind(
    "x",
    cl::init(InputKind::None),
    cl::desc("Input type"),
    cl::values(clEnumValN(InputKind::Verona, "verona", "Verona file")),
    cl::values(clEnumValN(InputKind::MLIR, "mlir", "MLIR file")));

  // Optimisations enabled
  static cl::opt<unsigned> optLevel(
    "O",
    cl::desc("Optimization level. [-O0, -O1, -O2, or -O3] "
             "(default = '-O0')"),
    cl::Prefix,
    cl::ZeroOrMore,
    cl::init(0));

  // Output file
  cl::opt<std::string> outputFile("o", cl::init(""), cl::desc("Output file"));

  // Grammar file is not optional
  std::string grammarFile;

  // Set defaults form command line arguments
  void cmdLineDefaults()
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
      llvm::StringRef filename(inputFile);
      if (filename == "-")
      {
        outputFile = "-";
      }
      else
      {
        std::string newName =
          filename.substr(0, filename.find_last_of('.')).str();
        if (inputKind == InputKind::MLIR)
          newName += ".new";
        newName += ".mlir";
        outputFile = newName;
      }
    }

    // Default grammar
    grammarFile = path::directory(path::executable()).append("/grammar.peg");
  }
} // namespace

int main(int argc, char** argv)
{
  // MLIR boilerplace
  mlir::registerAllDialects();
  mlir::registerAllPasses();
  // TODO: Register verona passes here.
  mlir::registerDialect<mlir::verona::VeronaDialect>();

  // Set up pretty-print signal handlers
  llvm::InitLLVM y(argc, argv);

  // Parse cmd-line options
  cl::ParseCommandLineOptions(argc, argv, "Verona MLIR Generator\n");
  cmdLineDefaults();

  if (inputKind == InputKind::None)
  {
    std::cerr << "ERROR: Unknown source type for '" << inputFile
              << "'. Must be [verona, mlir]" << std::endl;
    return 1;
  }

  // MLIR Generator
  mlir::verona::Generator gen;

  // Parse the source file (verona/mlir)
  switch (inputKind)
  {
    case InputKind::Verona:
    {
      // Parse the file
      err::Errors err;
      module::Passes passes = {sym::build, ref::build, prec::build};
      auto m = module::build(grammarFile, passes, inputFile, "verona", err);
      if (!err.empty())
      {
        std::cerr << "ERROR: cannot parse Verona file " << inputFile
                  << std::endl
                  << err.to_s() << std::endl;
        return 1;
      }
      // Parse AST file into MLIR
      try
      {
        gen.readAST(m->ast);
      }
      catch (std::runtime_error& e)
      {
        std::cerr << "ERROR: cannot convert Verona file " << inputFile
                  << " into MLIR" << std::endl
                  << e.what() << std::endl;
        return 1;
      }
      break;
    }
    case InputKind::MLIR:
      // Parse MLIR file
      try
      {
        gen.readMLIR(inputFile);
      }
      catch (std::runtime_error& e)
      {
        std::cerr << "ERROR: cannot read MLIR file " << inputFile << std::endl
                  << e.what() << std::endl;
        return 1;
      }
      break;
    default:
      std::cerr << "ERROR: invalid source file type" << std::endl;
      return 1;
  }

  // Emit MLIR
  try
  {
    gen.emitMLIR(outputFile, optLevel);
  }
  catch (std::runtime_error& e)
  {
    std::cerr << "ERROR: failed to lower to MLIR" << std::endl
              << e.what() << std::endl;
    return 1;
  }
  return 0;
}

// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "CLI/CLI.hpp"
#include "ast/cli.h"
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

#include "llvm/Support/InitLLVM.h"

namespace
{
  // Source file types, to choose how to parse
  enum class SourceKind
  {
    None,
    Verona,
    MLIR
  };

  // Command line options, with output control
  struct Opt
  {
    std::string grammar;
    std::string filename;
    std::string output;
    bool mlir = false;
    bool llvm = false;
  };

  // Parse cmd-line and set defaults
  Opt parse(int argc, char** argv)
  {
    CLI::App app{"Verona MLIR"};

    Opt opt;
    app.add_flag("--emit-mlir", opt.mlir, "Emit MLIR.");
    app.add_flag("--emit-llvm", opt.llvm, "Emit LLVM (default).");
    app.add_option("-g,--grammar", opt.grammar, "Grammar to use.");
    app.add_option("-o,--output", opt.output, "Output filename.");
    app.add_option("file", opt.filename, "File to compile.");

    try
    {
      app.parse(argc, argv);
    }
    catch (const CLI::ParseError& e)
    {
      exit(app.exit(e));
    }

    // Default is to output MLIR
    if (!opt.llvm)
      opt.mlir = true;

    // Default grammar
    if (opt.grammar.empty())
      opt.grammar = path::directory(path::executable()).append("/grammar.peg");

    // Default input is stdin
    if (opt.filename.empty())
      opt.filename = "-";

    return opt;
  }

  // Print help
  void help()
  {
    std::cout << "Compiler Syntax: verona-mlir AST|MLIR|LLVM <filename.verona>"
              << std::endl;
  }

  // Detect source type from extension
  SourceKind getSourceType(llvm::StringRef filename)
  {
    auto source = SourceKind::None;
    if (filename.endswith(".verona"))
      source = SourceKind::Verona;
    else if (filename.endswith(".mlir"))
      source = SourceKind::MLIR;
    else if (filename == "-") // STDIN, assume MLIR
      source = SourceKind::MLIR;
    return source;
  }

  // Choose output file extension from output type
  // Careful with mlir->mlir not to overwrite source file
  std::string
  getOutputFilename(llvm::StringRef filename, Opt& opt, SourceKind source)
  {
    if (!opt.output.empty())
      return opt.output;
    if (filename == "-")
      return "-";

    std::string newName = filename.substr(0, filename.find_last_of('.')).str();
    if (opt.mlir)
    {
      if (source == SourceKind::MLIR)
        newName += ".mlir.out";
      else
        newName += ".mlir";
    }
    else
    {
      newName += ".ll";
    }
    return newName;
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
  auto opt = parse(argc, argv);
  llvm::StringRef filename(opt.filename);
  auto source = getSourceType(filename);
  if (source == SourceKind::None)
  {
    std::cerr << "ERROR: Unknown source file " << filename.str()
              << ". Must be [verona, mlir]" << std::endl;
    return 1;
  }
  std::string outputFilename = getOutputFilename(filename, opt, source);

  // MLIR Generator
  mlir::verona::Generator gen;

  // Parse the source file (verona/mlir)
  switch (source)
  {
    case SourceKind::Verona:
    {
      // Parse the file
      err::Errors err;
      module::Passes passes = {sym::build, ref::build, prec::build};
      auto m = module::build(opt.grammar, passes, opt.filename, "verona", err);
      if (!err.empty())
      {
        std::cerr << "ERROR: cannot parse Verona file " << filename.str()
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
        std::cerr << "ERROR: cannot convert Verona file " << filename.str()
                  << " into MLIR" << std::endl
                  << e.what() << std::endl;
        return 1;
      }
      break;
    }
    case SourceKind::MLIR:
      // Parse MLIR file
      try
      {
        gen.readMLIR(opt.filename);
      }
      catch (std::runtime_error& e)
      {
        std::cerr << "ERROR: cannot read MLIR file " << filename.str()
                  << std::endl
                  << e.what() << std::endl;
        return 1;
      }
      break;
    default:
      std::cerr << "ERROR: invalid source file type" << std::endl;
      return 1;
  }

  // Emit the MLIR graph
  if (opt.mlir)
  {
    try
    {
      gen.emitMLIR(outputFilename);
    }
    catch (std::runtime_error& e)
    {
      std::cerr << "ERROR: failed to lower to MLIR" << std::endl
                << e.what() << std::endl;
      return 1;
    }
    return 0;
  }

  // Emit LLVM IR
  if (opt.llvm)
  {
    try
    {
      gen.emitLLVM(outputFilename);
    }
    catch (std::runtime_error& e)
    {
      std::cerr << "ERROR: failed to lower to LLVM" << std::endl
                << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
  return 0;
}

// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "ast/module.h"
#include "ast/parser.h"
#include "ast/path.h"
#include "ast/prec.h"
#include "ast/ref.h"
#include "ast/sym.h"
#include "driver.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"

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

  cl::opt<bool> splitInputFile(
    "split-input-file",
    cl::desc("Split the input file into pieces and "
             "process each chunk independently"),
    cl::init(false));

  cl::opt<bool> verifyDiagnostics(
    "verify-diagnostics",
    cl::desc("Check that emitted diagnostics match "
             "expected-* lines on the corresponding line"),
    cl::init(false));

  // FIXME: Move to llvm::sys::path, but LLVM's GetMainExecutable is horrible
  cl::opt<std::string> grammarFile(
    "grammar",
    cl::desc("Grammar file"),
    cl::value_desc("filename"),
    cl::init(path::directory(path::executable()).append("/grammar.peg")));

  /// Set default command-line arguments.
  ///
  /// Most defaults are configured using the cl::opt constructor. Some however
  /// depend on the value of other arguments. This function infers their value
  /// after all command-line arguments have been parsed.
  void inferCommandLineDefaults()
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

  /// Check the value of command-line arguments, beyond the verification
  /// provided by cl::opt.
  ///
  /// Prints an error message and returns a failure if an invalid combination is
  /// used.
  mlir::LogicalResult verifyCommandLine()
  {
    if (splitInputFile && inputKind != InputKind::MLIR)
    {
      llvm::errs() << "-split-input-file can only be used with MLIR inputs\n";
      return mlir::failure();
    }

    if (verifyDiagnostics && inputKind != InputKind::MLIR)
    {
      llvm::errs() << "-verify-diagnostics can only be used with MLIR inputs\n";
      return mlir::failure();
    }

    return mlir::success();
  }

  mlir::LogicalResult openInput(std::unique_ptr<llvm::MemoryBuffer>* input)
  {
    std::string err;
    *input = mlir::openInputFile(inputFile, &err);
    if (!*input)
    {
      llvm::errs() << err << "\n";
      return mlir::failure();
    }

    return mlir::success();
  }

  mlir::LogicalResult openOutput(std::unique_ptr<llvm::ToolOutputFile>* output)
  {
    std::string err;
    *output = mlir::openOutputFile(outputFile, &err);
    if (!*output)
    {
      llvm::errs() << err << "\n";
      return mlir::failure();
    }

    return mlir::success();
  }

  mlir::LogicalResult processVeronaInput(llvm::raw_ostream& output)
  {
    llvm::ExitOnError check;

    // Parse the file
    err::Errors errors;
    module::Passes passes = {sym::build, ref::build, prec::build};
    auto m = module::build(grammarFile, passes, inputFile, "verona", errors);
    if (!errors.empty())
    {
      llvm::errs() << "ERROR: cannot parse Verona file " << inputFile << "\n"
                   << errors.to_s() << "\n";
      return mlir::failure();
    }

    // Parse AST file into MLIR
    mlir::verona::Driver driver(optLevel);
    llvm::Error error = driver.readAST(m->ast);
    if (!error)
      error = driver.emitMLIR(output);

    if (!error)
      return mlir::success();

    driver.dumpMLIR(llvm::errs());
    logAllUnhandledErrors(std::move(error), llvm::errs());
    return mlir::failure();
  }

  // This function is called for each segment of the input file.
  // Usually there is only one segment, the entire file, but if
  // --split-input-file is used the file is split and this function is applied
  // to each part.
  mlir::LogicalResult processMLIRBuffer(
    std::unique_ptr<llvm::MemoryBuffer> buffer, llvm::raw_ostream& output)
  {
    mlir::verona::Driver driver(optLevel, verifyDiagnostics);

    llvm::Error error = driver.readMLIR(std::move(buffer));
    if (!error)
      error = driver.emitMLIR(output);

    // In verify-diagnostics mode, the driver will almost always fail, as
    // expected. We ignore its results and instead use the result from the
    // diagnostic handler.
    if (verifyDiagnostics)
    {
      // llvm::Error requires errors to be "handled", even if they were
      // expected, hence the empty handler.
      handleAllErrors(std::move(error), [](const llvm::ErrorInfoBase& e) {});
      error = driver.verifyDiagnostics();
    }

    if (!error)
      return mlir::success();

    driver.dumpMLIR(llvm::errs());
    logAllUnhandledErrors(std::move(error), llvm::errs());
    return mlir::failure();
  };

  mlir::LogicalResult processMLIRInput(llvm::raw_ostream& output)
  {
    std::unique_ptr<llvm::MemoryBuffer> input;
    if (mlir::failed(openInput(&input)))
      return mlir::failure();

    if (splitInputFile)
      return mlir::splitAndProcessBuffer(
        std::move(input), processMLIRBuffer, output);
    else
      return processMLIRBuffer(std::move(input), output);
  }
} // namespace

int main(int argc, char** argv)
{
  // Set up pretty-print signal handlers
  llvm::InitLLVM y(argc, argv);

  // Parse cmd-line options
  cl::ParseCommandLineOptions(argc, argv, "Verona MLIR Generator\n");
  inferCommandLineDefaults();
  if (mlir::failed(verifyCommandLine()))
    return 1;

  std::unique_ptr<llvm::ToolOutputFile> output;
  if (mlir::failed(openOutput(&output)))
    return 1;

  switch (inputKind)
  {
    case InputKind::Verona:
      if (mlir::failed(processVeronaInput(output->os())))
        return 1;
      break;

    case InputKind::MLIR:
      if (mlir::failed(processMLIRInput(output->os())))
        return 1;
      break;

    default:
      llvm::errs() << "ERROR: invalid source file type\n";
      return 1;
  }

  output->keep();
  return 0;
}

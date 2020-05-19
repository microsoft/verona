#include "cli.h"
#include "files.h"
#include "parser.h"
#include "sym.h"

// #include "llvm/ExecutionEngine/ExecutionEngine.h"
// #include "llvm/ExecutionEngine/GenericValue.h"
// #include "llvm/ExecutionEngine/MCJIT.h"
// #include <llvm/IR/IRBuilder.h>
// #include <llvm/IR/ValueSymbolTable.h>
// #include <llvm/IR/Verifier.h>
// #include <llvm/Support/TargetSelect.h>

int main(int argc, char** argv)
{
  auto opt = cli::parse(argc, argv);
  auto parser = parser::create(opt.grammar);
  auto ast = parser::parse(parser, opt.filename);

  if (!ast)
    return -1;

  err::Errors err;
  sym::build(ast, err);

  if (!err.empty())
  {
    std::cout << err.to_s() << std::endl;
    return -1;
  }

  // import
  // typecheck
  // codegen

  if (opt.ast)
    std::cout << peg::ast_to_s(ast) << std::endl;

  return 0;
}

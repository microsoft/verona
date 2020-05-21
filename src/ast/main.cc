// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "cli.h"
#include "files.h"
#include "parser.h"
#include "sym.h"

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

  if (opt.dump_path != "")
    files::write(opt.dump_path + "/ast.txt", peg::ast_to_s(ast));

  return 0;
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "cli.h"
#include "files.h"
#include "parser.h"
#include "prec.h"
#include "sym.h"

int main(int argc, char** argv)
{
  auto opt = cli::parse(argc, argv);
  auto parser = parser::create(opt.grammar);
  auto src = files::slurp(opt.filename);
  auto ast = parser::parse(parser, src, opt.filename);

  if (!ast)
    return -1;

  err::Errors err;
  sym::scope(ast, err);

  if (err.empty())
    sym::references(ast, err);

  if (err.empty())
    prec::build(ast, err);

  if (!err.empty())
  {
    std::cout << err.to_s() << std::endl;
    return -1;
  }

  if (opt.ast)
    std::cout << peg::ast_to_s(ast) << std::endl;

  return 0;
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "cli.h"
#include "files.h"
#include "parser.h"
#include "path.h"
#include "prec.h"
#include "ref.h"
#include "sym.h"

int main(int argc, char** argv)
{
  auto opt = cli::parse(argc, argv);
  auto parser = parser::create(opt.grammar);

  if (!parser)
  {
    std::cout << "Couldn't create parser from " << opt.grammar << std::endl;
    return -1;
  }

  auto ast = parser::parse(parser, opt.filename, "verona");

  if (!ast)
    return -1;

  err::Errors err;
  sym::build(ast, err);

  if (err.empty())
    ref::build(ast, err);

  if (err.empty())
    prec::build(ast, err);

  if (!err.empty())
  {
    std::cout << err.to_s() << std::endl;

    if (!opt.force)
      return -1;
  }

  if (opt.ast || opt.force)
    std::cout << peg::ast_to_s(ast) << std::endl;

  return 0;
}

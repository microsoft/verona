// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "cli.h"
#include "files.h"
#include "module.h"
#include "parser.h"
#include "path.h"
#include "prec.h"
#include "ref.h"
#include "sym.h"

int main(int argc, char** argv)
{
  err::Errors err;
  auto opt = cli::parse(argc, argv);
  auto parser = parser::create(opt.grammar, err);

  if (!err.empty())
  {
    std::cerr << err;
    return -1;
  }

  module::Passes passes = {sym::build, ref::build, prec::build};

  auto m = module::build(parser, passes, opt.filename, "verona", err);

  if (!err.empty())
    std::cerr << err;

  if (opt.ast)
    std::cout << m;

  return err.empty() ? 0 : -1;
}

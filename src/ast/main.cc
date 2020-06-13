// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "cli.h"
#include "module.h"
#include "prec.h"
#include "ref.h"
#include "sym.h"

int main(int argc, char** argv)
{
  module::Passes passes = {sym::build, ref::build, prec::build};
  err::Errors err;

  auto opt = cli::parse(argc, argv);
  auto m = module::build(opt.grammar, passes, opt.filename, "verona", err);

  if (!err.empty())
    std::cerr << err;

  if (opt.ast)
    std::cout << m;

  return err.empty() ? 0 : -1;
}

// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "parser.h"

int main(int argc, char** argv)
{
  err::Errors err;

  // auto opt = cli::parse(argc, argv);
  // auto m =
  //   module::build(opt.grammar, opt.stopAt, passes, opt.filename, "verona",
  //   err);

  // if (opt.ast)
  //   std::cout << m;

  std::cerr << err;
  return err.empty() ? 0 : -1;
}

// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "parser.h"
#include "print.h"

int main(int argc, char** argv)
{
  using namespace verona::parser;

  err::Errors err;

  // auto opt = cli::parse(argc, argv);
  // auto m =
  //   module::build(opt.grammar, opt.stopAt, passes, opt.filename, "verona",
  //   err);

  // if (opt.ast)
  //   std::cout << m;

  auto ast = parse("filename", err);
  std::cout << pretty(ast);

  std::cerr << err;
  return err.empty() ? 0 : -1;
}

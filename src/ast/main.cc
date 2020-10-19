// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "cli.h"
#include "module.h"
#include "pass.h"
#include "prec.h"
#include "ref.h"
#include "sym.h"

int main(int argc, char** argv)
{
  pass::Passes passes = {
    {"sym", sym::build}, {"ref", ref::build}, {"prec", prec::build}};
  err::Errors err;

  auto opt = cli::parse(argc, argv);
  auto m =
    module::build(opt.grammar, opt.stopAt, passes, opt.filename, "verona", err);

  if (opt.ast)
    std::cout << m;

  std::cerr << err;
  return err.empty() ? 0 : -1;
}

// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "ast/cli.h"
#include "ast/module.h"
#include "ast/pass.h"
#include "ast/prec.h"
#include "ast/ref.h"
#include "ast/sugar.h"
#include "ast/sym.h"
#include "typed-ast/conversion.h"
#include "typed-ast/print.h"

int main(int argc, char** argv)
{
  pass::Passes passes = {{"sugar", sugar::build},
                         {"sym", sym::build},
                         {"ref", ref::build},
                         {"prec", prec::build}};
  err::Errors err;

  auto opt = cli::parse(argc, argv);
  auto m =
    module::build(opt.grammar, opt.stopAt, passes, opt.filename, "verona", err);
  auto typedModule = verona::ast::convertModule(m->ast);

  if (opt.ast)
    verona::ast::print(std::cout, *typedModule);

  std::cerr << err;
  return err.empty() ? 0 : -1;
}

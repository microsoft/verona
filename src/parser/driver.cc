// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "driver.h"

#include "anf.h"
#include "dnf.h"
#include "infer.h"
#include "parser.h"
#include "path.h"
#include "print.h"
#include "resolve.h"

namespace verona::parser
{
  constexpr auto stdlib = "stdlib/";

  std::pair<bool, Ast> run(Limit limit, bool validate, const std::string& path)
  {
    auto stdlibpath = path::canonical(path::join(path::executable(), stdlib));
    auto [ok, ast] = parse(path, stdlibpath);
    ok = ok && (!validate || dnf::wellformed(ast));

    if (limit >= Limit::Resolve)
    {
      ok = ok && resolve::run(ast);
      ok = ok && (!validate || resolve::wellformed(ast));
    }

    if (limit >= Limit::Anf)
    {
      ok = ok && anf::run(ast);
      ok = ok && (!validate || anf::wellformed(ast));
    }

    if (limit >= Limit::Infer)
    {
      ok = ok && infer::run(ast);
      ok = ok && (!validate || infer::wellformed(ast));
    }

    return {ok, ast};
  }

  void dump(Ast& ast)
  {
    if (ast)
      std::cout << ast << std::endl;
  }
}

// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"

namespace verona::parser
{
  enum class Limit
  {
    Parse,
    Resolve,
    Anf,
    Infer,
  };

  std::pair<bool, Ast> run(Limit limit, bool validate, const std::string& path);

  void dump(Ast& ast);
}

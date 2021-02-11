// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"
#include "err.h"

#include <vector>

namespace pass
{
  /// A Pass object has a name and an operation, that is executed on AST nodes.
  class Pass
  {
    // The pass itself.
    using PassPtr = void (*)(ast::Ast& ast, err::Errors& err);
    PassPtr pass;

  public:
    Pass(const char* name, PassPtr pass) : pass(pass), name(name) {}

    /// The pass name, for debug purposes.
    const char* name;

    /// Runs the pass on an ast node.
    void operator()(ast::Ast& ast, err::Errors& err) const
    {
      pass(ast, err);
    }
  };

  /// A list of passes.
  using Passes = std::vector<Pass>;
}

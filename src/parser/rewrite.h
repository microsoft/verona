// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"

namespace verona::parser
{
  // This tries to replace the last node in the path with a new node. This will
  // succeed if the second to last node contains the last node in the path.
  bool rewrite(AstPath& path, size_t index, Ast node);

  // Clone this Ast.
  Ast clone(Substitutions& subs, Ast node);
}

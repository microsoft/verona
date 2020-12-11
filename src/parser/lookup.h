// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"

namespace verona::parser
{
  enum class Find
  {
    First,
    All,
  };

  AstPaths look_up(Find mode, AstPath& path, Location& name);

  AstPaths look_up(AstPath& path, List<TypeName>& names);
}

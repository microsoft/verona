// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"

namespace verona::parser::lookup
{
  Ast name(Ast symbols, const Location& name);

  Ast typenames(Ast symbols, List<TypeName>& names);

  void reset(List<TypeName>& names);
}

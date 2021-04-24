// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"

#include <iostream>

namespace verona::parser
{
  struct Lookup
  {
    std::function<std::ostream&(void)> error;

    Lookup(std::function<std::ostream&(void)> error) : error(error) {}

    // This sets the context, def, and subs map for the typeref.
    Ast typeref(Ast sym, TypeRef& tr);

    // This looks up `name` in the lexical scope, following `using` statements.
    Ast name(Ast& sym, const Location& name);

    // This looks up `name` as a member of `node`. If `node` is not a Class or
    // an Interface, `node` is first resolved in the symbol context. This is
    // used for static lookup, so it fails on union types or when an
    // intersection contains more than one option.
    Ast member(Ast& sym, Ast node, const Location& name);

    // This adds typeargs to the substitution map.
    void substitutions(Substitutions& subs, Ast& def, List<Type>& typeargs);
  };
}

// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "subtype.h"

#include <iostream>

namespace verona::parser
{
  struct Lookup
  {
    std::function<std::ostream&(void)> error;
    Subtype::BoundsMap* bounds;

    Lookup(
      std::function<std::ostream&(void)> error,
      Subtype::BoundsMap* bounds = nullptr)
    : error(error), bounds(bounds)
    {}

    // This sets the context, def, and subs map for the typeref. The def is
    // returned, and will be a Class, Interface, TypeAlias, Field, Function,
    // LookupUnion, or LookupIsect.
    Ast typeref(Ast sym, TypeRef& tr);

    // This looks up `name` in the lexical scope, following `using` statements.
    // The result will be a Class, Interface, TypeAlias, Field, Function,
    // TypeParam, Param, Let, or Var. The closest one will be returned, where a
    // `using` statement is the furthest away in its scope, but closer than the
    // enclosing scope.
    Ast name(Ast& sym, const Location& name);

    // This looks up `name` as a member of `node`. If `node` is not a Class or
    // an Interface, `node` is first resolved in the symbol context. The result
    // will be a Class, Interface, TypeAlias, Field, Function, LookupUnion, or
    // LookupIsect.
    Ast member(Ast& sym, Ast node, const Location& name);

    // This adds typeargs to the substitution map.
    void substitutions(Substitutions& subs, Ast& def, List<Type>& typeargs);

    // Helper functions for looking up in union and isect types.
    Ast union_member(Ast& sym, List<Type>& list, const Location& name);
    Ast isect_member(Ast& sym, List<Type>& list, const Location& name);

    // DNF over lookup values.
    Ast disjunction(Ast left, Ast right);
    Ast conjunction(Ast left, Ast right);
  };
}

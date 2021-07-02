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
    Node<LookupResult> typeref(Ast sym, TypeRef& tr);
    Node<LookupResult> typeref(Substitutions& subs, Ast sym, TypeRef& tr);

    // This looks up `name` in the lexical scope, following `using` statements.
    // The result will be a Class, Interface, TypeAlias, Field, Function,
    // TypeParam, Param, Let, or Var. The closest one will be returned, where a
    // `using` statement is the furthest away in its scope, but closer than the
    // enclosing scope.
    Node<LookupResult> name(Substitutions& subs, Ast sym, Node<TypeName>& tn);

    // This looks up `name` as a member of `node`. If `node` is not a Class or
    // an Interface, `node` is first resolved in the symbol context. The result
    // will be a Class, Interface, TypeAlias, Field, Function, LookupUnion, or
    // LookupIsect.
    Node<LookupResult> member(Ast node, Node<TypeName>& tn);
    Node<LookupResult>
    member(Substitutions& subs, Ast sym, Ast node, Node<TypeName>& tn);

    // This adds typeargs to the substitution map.
    Substitutions
    substitutions(Substitutions& subs, Ast def, List<Type>& typeargs);
  };
}

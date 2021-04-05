// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lookup.h"

#include <iostream>
#include <unordered_set>

namespace verona::parser::lookup
{
  // This looks up `name` as a member of `node`. If `node` is not a Class or an
  // Interface, `node` is first resolved in the symbol context. If this
  Ast member(Ast& symbols, Ast node, const Location& name)
  {
    switch (node->kind())
    {
      case Kind::Class:
      case Kind::Interface:
      {
        // Update the symbol context.
        symbols = node;

        // Look in the symbol table.
        auto def = node->symbol_table()->get(name);

        if (def)
          return def;

        return {};
      }

      case Kind::TypeAlias:
      {
        // Look in the type we are aliasing.
        return member(symbols, node->as<TypeAlias>().inherits, name);
      }

      case Kind::TypeParam:
      {
        // Look in our upper bounds.
        return member(symbols, node->as<TypeParam>().upper, name);
      }

      case Kind::ExtractType:
      case Kind::ViewType:
      {
        // This is the result of a `using`, a type alias, or a type parameter.
        // Look in the right-hand side.
        return member(symbols, node->as<TypePair>().right, name);
      }

      case Kind::TypeRef:
      {
        // This is the result of a `using`, a type alias, or a type parameter.
        // Look in the resolved type.
        auto def = typenames(symbols, node->as<TypeRef>().typenames);

        // Update the symbol context.
        if (is_kind(def, {Kind::Class, Kind::Interface, Kind::TypeAlias}))
          symbols = def;

        return member(symbols, def, name);
      }

      case Kind::IsectType:
      {
        // Look in all conjunctions.
        // TODO: what if we find it more than once?
        auto& isect = node->as<IsectType>();

        for (auto& type : isect.types)
        {
          auto def = member(symbols, type, name);

          if (def)
            return def;
        }

        return {};
      }

      case Kind::UnionType:
      {
        // Look in all disjunctions.
        // TODO: must be present everywhere
        return {};
      }

      default:
      {
        // No lookup in Field, Function, ThrowType, FunctionType, TupleType,
        // TypeList, or a capability.
        // TODO: Self
        return {};
      }
    }
  }

  Ast name(Ast symbols, const Location& name)
  {
    while (symbols)
    {
      auto st = symbols->symbol_table();
      assert(st != nullptr);

      auto def = st->get(name);

      if (def)
        return def;

      for (auto it = st->use.rbegin(); it != st->use.rend(); ++it)
      {
        auto& use = *it;

        // Only accept `using` statements in the same file.
        if (use->location.source->origin != name.source->origin)
          continue;

        // Only accept `using` statements that are earlier in scope.
        if (use->location.start > name.start)
          continue;

        // Find `name` in the used TypeRef, using the current symbol table.
        def = member(symbols, use->type, name);

        if (def)
          return def;
      }

      symbols = st->parent.lock();
    }

    return {};
  }

  Ast typenames(Ast symbols, List<TypeName>& names)
  {
    // Each element will have a definition. This will point to a Class,
    // Interface, TypeAlias, Field, or Function.

    // This shouldn't happen.
    if (names.empty())
      return {};

    // Return the cached lookup.
    auto def = names.back()->def.lock();

    if (def)
      return def;

    // Check if we failed previously and only partially resolved this.
    def = names.front()->def.lock();

    if (def)
      return {};

    // Lookup the first element in the current symbol context.
    def = name(symbols, names.front()->location);

    if (!def)
      return {};

    names.front()->def = def;

    for (size_t i = 1; i < names.size(); i++)
    {
      // Look in the current definition for the next name.
      def = member(symbols, def, names.at(i)->location);
      names.at(i)->def = def;
    }

    return def;
  }

  void reset(List<TypeName>& names)
  {
    for (auto& n : names)
      n->def.reset();
  }
}

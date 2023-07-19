// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lookup.h"

#include "lang.h"

#include <cassert>

namespace verona
{
  Lookup::Lookup(Node def, Node ta, NodeMap<Node> b) : def(def), bindings(b)
  {
    if (!def->type().in({Class, TypeAlias, Function}))
      return;

    auto tp = def / TypeParams;
    size_t n = ta ? ta->size() : 0;

    if (tp->size() < n)
      n = tp->size();

    if (n > 0)
    {
      // Bind the first `n` typeparams to the first `n` typeargs.
      std::transform(
        ta->begin(),
        ta->begin() + n,
        tp->begin(),
        std::inserter(bindings, bindings.end()),
        [](auto arg, auto param) { return std::make_pair(param, arg); });
    }

    // Bind all remaining typeparams to fresh typevars.
    std::transform(
      tp->begin() + n,
      tp->end(),
      std::inserter(bindings, bindings.end()),
      [](auto param) {
        return std::make_pair(param, TypeVar ^ param->fresh());
      });
  }

  Lookups lookdown_all(Node tn, Node id, Node ta, NodeSet visited);

  Lookups lookdown(Lookup& lookup, Node id, Node ta, NodeSet visited)
  {
    while (true)
    {
      // Check if we've visited this node before. If so, we've found a cycle.
      auto inserted = visited.insert(lookup.def);
      if (!inserted.second)
        return {};

      if (lookup.def->type().in({Class, TypeTrait, Function}))
      {
        // Return all lookdowns in the found class, trait, or function.
        Lookups result;
        auto defs = lookup.def->lookdown(id->location());

        std::transform(
          defs.cbegin(),
          defs.cend(),
          std::back_inserter(result),
          [&](auto& def) { return Lookup(def, ta, lookup.bindings); });

        return result;
      }
      else if (lookup.def->type() == TypeAlias)
      {
        // Replace the def with our type alias and try again.
        lookup.def = lookup.def / Type;
      }
      else if (lookup.def->type() == TypeParam)
      {
        // Replace the typeparam with the bound typearg and try again.
        auto it = lookup.bindings.find(lookup.def);

        if ((it != lookup.bindings.end()) && it->second)
          lookup.def = it->second;
        else
          return {};
      }
      // The remainder of cases arise from a Use, a TypeAlias, or a TypeParam.
      // They will all result in some number of name resolutions.
      else if (lookup.def->type() == Type)
      {
        // Replace the def with the content of the type and try again.
        // Use `front()` instead of `def / Type` to allow looking up in `use`
        // directives before Type is no longer a sequence.
        lookup.def = lookup.def->front();
      }
      else if (lookup.def->type().in(
                 {TypeClassName, TypeAliasName, TypeTraitName, TypeParamName}))
      {
        // Resolve the name and try again. Pass `visited` into the resulting
        // lookdowns, so that each path tracks cycles independently.
        return lookdown_all(lookup.def, id, ta, visited);
      }
      else if (lookup.def->type() == TypeView)
      {
        // Replace the def with the rhs of the view and try again.
        lookup.def = lookup.def->back();
      }
      else if (lookup.def->type() == TypeIsect)
      {
        // Return everything we find.
        Lookups result;

        for (auto& t : *lookup.def)
        {
          auto l = Lookup(t, lookup.bindings);
          auto ldefs = lookdown(l, id, ta, visited);
          result.insert(result.end(), ldefs.begin(), ldefs.end());
        }

        return result;
      }
      else if (lookup.def->type() == TypeUnion)
      {
        // TODO: return only things that are identical in all disjunctions
        return {};
      }
      else if (lookup.def->type().in({TypeList, TypeTuple, TypeVar}))
      {
        // Nothing to do here.
        return {};
      }
      else
      {
        // This type isn't resolved yet.
        return {};
      }
    }
  }

  Lookups lookdown_all(Node tn, Node id, Node ta, NodeSet visited)
  {
    auto defs = lookup_scopedname(tn);
    Lookups result;

    for (auto& def : defs)
    {
      auto ldefs = lookdown(def, id, ta, visited);
      result.insert(result.end(), ldefs.begin(), ldefs.end());
    }

    return result;
  }

  Lookups lookup_name(Node id, Node ta)
  {
    assert(id->type().in({Ident, Symbol}));
    assert(!ta || (ta->type() == TypeArgs));

    auto defs = id->lookup();
    Lookups result;

    for (auto& def : defs)
    {
      if (def->type() == Use)
      {
        // Expand Use nodes by looking down into the target type.
        if (def->precedes(id))
        {
          auto l = Lookup(def / Type);
          auto ldefs = lookdown(l, id, ta, {});
          result.insert(result.end(), ldefs.begin(), ldefs.end());
        }
      }
      else
      {
        result.emplace_back(def, ta);
      }
    }

    return result;
  }

  Lookups lookup_scopedname(Node tn)
  {
    if (tn->type() == Error)
      return {};

    assert(tn->type().in(
      {TypeClassName,
       TypeAliasName,
       TypeParamName,
       TypeTraitName,
       FunctionName}));

    return lookup_scopedname_name(tn / Lhs, tn / Ident, tn / TypeArgs);
  }

  Lookups lookup_scopedname_name(Node tn, Node id, Node ta)
  {
    if (tn->type() == DontCare)
      return lookup_name(id, ta);

    return lookdown_all(tn, id, ta, {});
  }

  bool lookup(const NodeRange& n, std::initializer_list<Token> t)
  {
    auto defs = lookup_name(*n.first, {});
    return (defs.size() == 1) && defs.front().def->type().in(t);
  }
}

// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lookup.h"

#include "lang.h"

#include <cassert>
#include <deque>

namespace verona
{
  Lookup::Lookup(Node def, Node ta, NodeMap<Node> b) : def(def), bindings(b)
  {
    if (!def->type().in({Class, TypeAlias, Function}))
    {
      if (ta && !ta->empty())
        too_many_typeargs = true;

      return;
    }

    if (!ta)
      return;

    auto tp = def / TypeParams;

    if (tp->size() < ta->size())
    {
      too_many_typeargs = true;
      return;
    }

    // Bind all typeparams to their corresponding typeargs.
    std::transform(
      ta->begin(),
      ta->end(),
      tp->begin(),
      std::inserter(bindings, bindings.end()),
      [](auto arg, auto param) { return std::make_pair(param, arg); });

    // Bind all remaining typeparams to fresh typevars.
    std::transform(
      tp->begin() + ta->size(),
      tp->end(),
      std::inserter(bindings, bindings.end()),
      [](auto param) {
        return std::make_pair(param, TypeVar ^ param->fresh());
      });
  }

  Lookups lookdown(Lookups&& lookups, Node id, Node ta, NodeSet visited = {});

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
          std::back_inserter(result.defs),
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
        return lookdown(lookup_scopedname(lookup.def), id, ta, visited);
      }
      else if (lookup.def->type() == TypeView)
      {
        // Replace the def with the rhs of the view and try again.
        lookup.def = lookup.def->back();
      }
      else if (lookup.def->type() == TypeIsect)
      {
        // TODO: return everything we find
        return {};
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

  Lookups lookdown(Lookups&& lookups, Node id, Node ta, NodeSet visited)
  {
    Lookups result;

    for (auto& lookup : lookups.defs)
      result.add(lookdown(lookup, id, ta, visited));

    return result;
  }

  Lookups lookup_name(Node id, Node ta)
  {
    assert(id->type().in({Ident, Symbol}));
    assert(!ta || (ta->type() == TypeArgs));

    Lookups lookups;
    auto defs = id->lookup();

    for (auto& def : defs)
    {
      if (def->type() == Use)
      {
        // Expand Use nodes by looking down into the target type.
        if (def->precedes(id))
          lookups.add(lookdown(Lookup(def / Type), id, ta));
      }
      else
      {
        lookups.add(Lookup(def, ta));
      }
    }

    return lookups;
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

    return lookdown(lookup_scopedname(tn), id, ta);
  }

  bool lookup(const NodeRange& n, std::initializer_list<Token> t)
  {
    return lookup_name(*n.first).one(t);
  }

  bool recursive_typealias(Node node)
  {
    // This detects cycles in type aliases, which are not allowed.
    if (node->type() != TypeAlias)
      return false;

    // Each element in the worklist carries a set of nodes that have been
    // visited, a type node, and a map of typeparam bindings.
    std::deque<std::pair<NodeSet, Lookup>> worklist;
    worklist.emplace_back(NodeSet{node}, node / Type);

    while (!worklist.empty())
    {
      auto& work = worklist.front();
      auto& set = work.first;
      auto& type = work.second.def;
      auto& bindings = work.second.bindings;

      if (type->type() == Type)
      {
        worklist.emplace_back(set, Lookup(type / Type, bindings));
      }
      else if (type->type().in({TypeTuple, TypeUnion, TypeIsect, TypeView}))
      {
        for (auto& t : *type)
          worklist.emplace_back(set, Lookup(t, bindings));
      }
      else if (type->type() == TypeAliasName)
      {
        auto defs = lookup_scopedname(type);

        if (!defs.defs.empty())
        {
          auto& def = defs.defs.front();

          if (set.contains(def.def))
            return true;

          for (auto& bind : def.bindings)
            bindings[bind.first] = bind.second;

          set.insert(def.def);
          worklist.emplace_back(set, Lookup(def.def / Type, bindings));
        }
      }
      else if (type->type() == TypeParamName)
      {
        auto defs = lookup_scopedname(type);

        if (!defs.defs.empty())
        {
          auto& def = defs.defs.front();
          auto find = bindings.find(def.def);

          if (find != bindings.end())
            worklist.emplace_back(set, Lookup(find->second, bindings));
        }
      }

      worklist.pop_front();
    }

    return false;
  }
}

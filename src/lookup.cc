// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lookup.h"

#include "lang.h"

#include <cassert>
#include <charconv>

namespace verona
{
  size_t Lookup::sub(Node& node)
  {
    // Substitutes inside of `node`, but not `node` itself.
    size_t changes = 0;

    for (auto child : *node)
    {
      while ((child == FQType) && ((child / Type) == TypeParamName))
      {
        auto l = resolve_fq(child);
        auto it = bindings.find(l.def);

        if (it == bindings.end())
          break;

        auto bind = clone(it->second);

        if (bind == Type)
          bind = bind / Type;

        if ((bind == FQType) && ((bind / Type) == TypeParamName))
        {
          auto l_bind = resolve_fq(bind);

          if (l.def == l_bind.def)
            break;
        }

        node->replace(child, bind);
        child = bind;
      }

      changes += sub(child);
    }

    return changes;
  }

  static void apply_typeargs(Lookup& lookup, Node ta)
  {
    if (!lookup.def->in({Class, TypeAlias, Function}))
      return;

    auto tp = lookup.def / TypeParams;
    size_t n = ta ? ta->size() : 0;

    if (tp->size() < n)
    {
      lookup.too_many_typeargs = true;
      n = tp->size();
    }

    if (n > 0)
    {
      // Bind the first `n` typeparams to the first `n` typeargs.
      std::transform(
        ta->begin(),
        ta->begin() + n,
        tp->begin(),
        std::inserter(lookup.bindings, lookup.bindings.end()),
        [](auto arg, auto param) { return std::make_pair(param, arg); });
    }

    // Bind all remaining typeparams to fresh typevars.
    std::transform(
      tp->begin() + n,
      tp->end(),
      std::inserter(lookup.bindings, lookup.bindings.end()),
      [](auto param) { return std::make_pair(param, typevar(param)); });
  }

  Lookups lookup(Node id, Node ta)
  {
    assert(id->in({Ident, Symbol, Self}));
    assert(!ta || (ta == TypeArgs));
    Lookups result;

    if (id == Self)
    {
      auto def = id->parent({Class, Trait});
      Lookup l(def);
      apply_typeargs(l, ta);
      result.emplace(l);
      return result;
    }

    auto defs = id->lookup();

    for (auto& def : defs)
    {
      if (def == Use)
      {
        // Expand Use nodes by looking down into the target type.
        if (def->precedes(id))
        {
          auto l = Lookup(def / Type);
          auto ldefs = lookdown(l, id, ta, {});
          result.insert(ldefs.begin(), ldefs.end());
        }
      }
      else
      {
        Lookup l(def);
        apply_typeargs(l, ta);
        result.emplace(l);
      }
    }

    return result;
  }

  Lookups lookdown(Lookup& lookup, Node id, Node ta, NodeSet visited)
  {
    while (true)
    {
      // Stop on a failed lookup.
      if (!lookup.def)
        return {};

      // Check if we've visited this node before. If so, we've found a cycle.
      auto inserted = visited.insert(lookup.def);
      if (!inserted.second)
        return {};

      if (lookup.def->in({Class, Trait, Function}))
      {
        // Return all lookdowns in the found class, trait, or function.
        Lookups result;
        auto defs = lookup.def->lookdown(id->location());

        std::transform(
          defs.begin(),
          defs.end(),
          std::inserter(result, result.begin()),
          [&](auto& def) {
            auto l = lookup.make(def);
            apply_typeargs(l, ta);
            return l;
          });

        return result;
      }
      else if (lookup.def == TypeAlias)
      {
        // Replace the def with our type alias and try again.
        lookup.def = lookup.def / Type;
      }
      else if (lookup.def == TypeParam)
      {
        // Replace the typeparam with the bound typearg and try again.
        auto it = lookup.bindings.find(lookup.def);

        // Except in testing, there should always be a binding. If it's bound
        // to itself, then we're done.
        if ((it == lookup.bindings.end()) || (it->second == TypeParamBind))
          return {};

        lookup.def = it->second;
      }
      // The remainder of cases arise from a Use, a TypeAlias, or a TypeParam.
      // They will all result in some number of name resolutions.
      else if (lookup.def == Type)
      {
        // Replace the def with the content of the type and try again.
        // Use `front()` instead of `def / Type` to allow looking up in `use`
        // directives before Type is no longer a sequence.
        lookup.def = lookup.def->front();
      }
      else if (lookup.def->in({FQType, FQFunction}))
      {
        // Resolve the name and try again.
        lookup = resolve_fq(lookup.def);
      }
      else if (lookup.def == TypeView)
      {
        // Replace the def with the rhs of the view and try again.
        lookup.def = lookup.def->back();
      }
      else if (lookup.def == TypeIsect)
      {
        // Return everything we find.
        Lookups result;

        for (auto& t : *lookup.def)
        {
          auto l = lookup.make(t);
          auto ldefs = lookdown(l, id, ta, visited);
          result.insert(ldefs.begin(), ldefs.end());
        }

        return result;
      }
      else if (lookup.def == TypeUnion)
      {
        // TODO: return only things that are identical in all disjunctions
        return {};
      }
      else if (lookup.def->in({TypeList, TypeTuple, TypeVar}))
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

  bool lookup_type(Node id, std::initializer_list<Token> t)
  {
    auto defs = lookup(id, {});
    return (defs.size() == 1) && defs.begin()->def->in(t);
  }

  bool lookup_type(const NodeRange& n, std::initializer_list<Token> t)
  {
    return lookup_type(*n.first, t);
  }

  static bool resolve_selector(Lookup& p, Node n)
  {
    assert(n == Selector);

    if (!p.def)
      return false;

    auto hand = (n / Ref)->type();
    auto id = n / Ident;
    auto defs = p.def->lookdown(id->location());

    size_t arity = 0;
    auto view = (n / Int)->location().view();
    std::from_chars(view.data(), view.data() + view.size(), arity);

    for (auto& def : defs)
    {
      if (
        (def == Function) && ((def / Ref) == hand) &&
        ((def / Params)->size() == arity))
      {
        p = p.make(def);
        apply_typeargs(p, n / TypeArgs);
        return true;
      }
    }

    return false;
  }

  static bool resolve_typename(Lookup& p, Node n)
  {
    assert(n->in({TypeClassName, TypeAliasName, TypeParamName, TypeTraitName}));
    auto def = p.def;

    if (!def)
      return false;

    auto id = n / Ident;
    auto defs = def->lookdown(id->location());

    if (defs.size() != 1)
      return false;

    p = p.make(*defs.begin());

    if (n->in({TypeClassName, TypeAliasName}))
      apply_typeargs(p, n / TypeArgs);

    return true;
  }

  static Lookup resolve_fqtype(Node fq)
  {
    assert(fq == FQType);
    Lookup p(ast::top());
    auto path = fq / TypePath;

    for (auto& n : *path)
    {
      if (n == Selector)
      {
        if (!resolve_selector(p, n))
          return {};
      }
      else if (n->in(
                 {TypeClassName, TypeAliasName, TypeParamName, TypeTraitName}))
      {
        if (!resolve_typename(p, n))
          return {};
      }
    }

    if (!resolve_typename(p, fq / Type))
      return {};

    return p;
  }

  Lookup resolve_fq(Node fq)
  {
    if (fq == FQType)
      return resolve_fqtype(fq);

    assert(fq == FQFunction);
    auto p = resolve_fqtype(fq / FQType);

    if (!resolve_selector(p, fq / Selector))
      return {};

    return p;
  }

  static Node make_typeargs(Node& node, Lookup& lookup, bool fresh)
  {
    auto tps = node / TypeParams;
    Node ta = TypeArgs;

    for (auto& tp : *tps)
    {
      auto it = lookup.bindings.find(tp);

      if (it != lookup.bindings.end())
        ta << clone(it->second);
      else if (fresh)
        ta << typevar(node);
      else
        ta << TypeParamBind;
    }

    return ta;
  }

  static Node make_fq(Lookup& lookup, bool fresh)
  {
    if (!lookup.def->in({Class, TypeAlias, TypeParam, Trait, Function}))
      return lookup.def;

    Node path = TypePath;
    auto node = lookup.def;

    while (node)
    {
      if (node == Class)
      {
        path
          << (TypeClassName << clone(node / Ident)
                            << make_typeargs(node, lookup, fresh));
      }
      else if (node == TypeAlias)
      {
        path
          << (TypeAliasName << clone(node / Ident)
                            << make_typeargs(node, lookup, fresh));
      }
      else if (node == TypeParam)
      {
        path << (TypeParamName << clone(node / Ident));
      }
      else if (node == Trait)
      {
        path << (TypeTraitName << clone(node / Ident));
      }
      else if (node == Function)
      {
        auto arity = std::to_string((node / Params)->size());

        path
          << (Selector << clone(node / Ref) << clone(node / Ident)
                       << (Int ^ arity) << make_typeargs(node, lookup, fresh));
      }

      node = node->parent({Class, TypeAlias, Trait, Function});
    }

    std::reverse(path->begin(), path->end());
    node = path->pop_back();

    if (node->in({TypeClassName, TypeAliasName, TypeParamName, TypeTraitName}))
      return FQType << path << node;

    assert(!path->empty());
    auto t = path->pop_back();
    return FQFunction << (FQType << path << t) << node;
  }

  Node make_fq(Lookup& lookup)
  {
    return make_fq(lookup, false);
  }

  Node local_fq(Node node)
  {
    // Build an FQType for the local type.
    if (!node->in({Class, TypeAlias, TypeParam, Trait, Function}))
      node = node->parent({Class, TypeAlias, TypeParam, Trait, Function});

    Lookup l(node);
    return make_fq(l, true);
  }

  Node append_fq(Node fq, Node node)
  {
    assert(fq->type().in({FQType, FQFunction}));
    assert(node->in(
      {TypeClassName, TypeAliasName, TypeParamName, TypeTraitName, Selector}));

    if (fq == FQFunction)
    {
      // FQFunction + Selector = not allowed
      assert(node != Selector);

      // FQFunction + Type = FQType
      return FQType << (clone(fq / FQType / TypePath)
                        << clone(fq / FQType / Type) << clone(fq / Selector))
                    << node;
    }

    // FQType + Selector = FQFunction
    if (node == Selector)
      return FQFunction << clone(fq) << node;

    // FQType + Type = FQType
    return FQType << (clone(fq / TypePath) << clone(fq / Type)) << node;
  }
}

// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../btype.h"

namespace verona
{
  bool recursive_typealias(Node node)
  {
    // This detects cycles in type aliases, which are not allowed.
    if (node->type() != TypeAlias)
      return false;

    // Each element in the worklist carries a set of nodes that have been
    // visited, a type node, and a map of typeparam bindings.
    std::vector<std::pair<NodeSet, Lookup>> worklist;
    worklist.emplace_back(NodeSet{node}, node / Type);

    while (!worklist.empty())
    {
      auto work = worklist.back();
      auto& set = work.first;
      auto& type = work.second.def;
      auto& bindings = work.second.bindings;
      worklist.pop_back();

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

        if (!defs.empty())
        {
          auto& def = defs.front();

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

        if (!defs.empty())
        {
          auto& def = defs.front();
          auto find = bindings.find(def.def);

          if (find != bindings.end())
            worklist.emplace_back(set, Lookup(find->second, bindings));
        }
      }
    }

    return false;
  }

  bool valid_predicate(Btype t)
  {
    // A predicate is a type that can be used in a where clause. They can be
    // composed of unions and intersections of predicates and type aliases
    // that expand to predicates.
    std::vector<Btype> worklist;
    worklist.push_back(t);

    while (!worklist.empty())
    {
      t = worklist.back();
      worklist.pop_back();

      if (t->type() == TypeSubtype)
      {
        // Do nothing.
      }
      else if (t->type().in({TypeUnion, TypeIsect}))
      {
        // Check that all children are valid predicates.
        std::for_each(t->node->begin(), t->node->end(), [&](auto& n) {
          worklist.push_back(t->make(n));
        });
      }
      else if (t->type() == TypeAlias)
      {
        worklist.push_back(t->field(Type));
      }
      else
      {
        return false;
      }
    }

    return true;
  }

  bool valid_inherit(Btype t)
  {
    // A type that can be used in an inherit clause. They can be composed of
    // intersections of classes, traits, and type aliases that expand to
    // valid inherit clauses.
    std::vector<Btype> worklist;
    worklist.push_back(t);

    while (!worklist.empty())
    {
      t = worklist.back();
      worklist.pop_back();

      if (t->type().in({Class, TypeTrait}))
      {
        // Do nothing.
      }
      else if (t->type().in({Type, TypeIsect}))
      {
        // Check that all children are valid for code reuse.
        std::for_each(t->node->begin(), t->node->end(), [&](auto& n) {
          worklist.push_back(t->make(n));
        });
      }
      else if (t->type() == TypeAlias)
      {
        worklist.push_back(t->field(Type));
      }
      else
      {
        return false;
      }
    }

    return true;
  }

  PassDef typevalid()
  {
    return {
      dir::once | dir::topdown,
      {
        T(TypeAlias)[TypeAlias] >> ([](Match& _) -> Node {
          if (recursive_typealias(_(TypeAlias)))
            return err(_[TypeAlias], "recursive type alias");

          return NoChange;
        }),

        In(TypePred)++ * --(In(TypeSubtype, TypeArgs)++) *
            T(TypeAliasName)[TypeAliasName] >>
          ([](Match& _) -> Node {
            if (!valid_predicate(make_btype(_(TypeAliasName))))
              return err(
                _[Type], "this type alias isn't a valid type predicate");

            return NoChange;
          }),

        In(TypePred)++ * --(In(TypeSubtype, TypeArgs)++) *
            (TypeCaps / T(TypeClassName) / T(TypeParamName) / T(TypeTraitName) /
             T(TypeTrait) / T(TypeTuple) / T(Self) / T(TypeList) / T(TypeView) /
             T(TypeVar) / T(Package))[Type] >>
          [](Match& _) {
            return err(_[Type], "can't put this in a type predicate");
          },

        In(Inherit)++ * --(In(TypeArgs)++) * T(TypeAliasName)[TypeAliasName] >>
          ([](Match& _) -> Node {
            if (!valid_inherit(make_btype(_(TypeAliasName))))
              return err(
                _[Type], "this type alias isn't valid for inheritance");

            return NoChange;
          }),

        In(Inherit)++ * --(In(TypeArgs)++) *
            (TypeCaps / T(TypeParamName) / T(TypeTuple) / T(Self) /
             T(TypeList) / T(TypeView) / T(TypeUnion) / T(TypeVar) /
             T(Package) / T(TypeSubtype) / T(TypeTrue) / T(TypeFalse))[Type] >>
          [](Match& _) { return err(_[Type], "can't inherit from this type"); },
      }};
  }
}

// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../btype.h"

namespace verona
{
  bool recursive_typealias(Node node)
  {
    // This detects cycles in type aliases, which are not allowed. This happens
    // after type names are turned into FQType.
    assert(node->type() == TypeAlias);

    // Each element in the worklist carries a set of nodes that have been
    // visited, a type node, and a map of typeparam bindings.
    std::vector<std::pair<NodeSet, Lookup>> worklist;
    worklist.emplace_back(NodeSet{node}, node / Type);

    while (!worklist.empty())
    {
      auto work = worklist.back();
      auto& set = work.first;
      auto& lookup = work.second;
      worklist.pop_back();

      if (lookup.def->type() == Type)
      {
        worklist.emplace_back(set, lookup.make(lookup.def / Type));
      }
      else if (lookup.def->type().in(
                 {TypeTuple, TypeUnion, TypeIsect, TypeView}))
      {
        for (auto& t : *lookup.def)
          worklist.emplace_back(set, lookup.make(t));
      }
      else if (
        (lookup.def->type() == FQType) &&
        ((lookup.def / Type)->type() == TypeAliasName))
      {
        auto l = resolve_fq(lookup.def);

        if (l.def)
        {
          if (set.contains(l.def))
            return true;

          set.insert(l.def);
          worklist.emplace_back(set, l);
        }
      }
      else if (
        (lookup.def->type() == FQType) &&
        ((lookup.def / Type)->type() == TypeParamName))
      {
        auto l = resolve_fq(lookup.def);

        if (l.def)
        {
          auto find = lookup.bindings.find(l.def);

          if (find != lookup.bindings.end())
            worklist.emplace_back(set, lookup.make(find->second));
        }
      }
    }

    return false;
  }

  bool recursive_inherit(Node node)
  {
    assert(node->type() == Inherit);
    std::vector<std::pair<NodeSet, Lookup>> worklist;
    worklist.emplace_back(NodeSet{node}, node / Inherit);

    while (!worklist.empty())
    {
      auto work = worklist.back();
      auto& set = work.first;
      auto& lookup = work.second;
      worklist.pop_back();

      if (lookup.def->type() == Type)
      {
        worklist.emplace_back(set, lookup.make(lookup.def / Type));
      }
      else if (lookup.def->type() == TypeIsect)
      {
        for (auto& t : *lookup.def)
          worklist.emplace_back(set, lookup.make(t));
      }
      else if (
        (lookup.def->type() == FQType) &&
        ((lookup.def / Type)->type() == TypeClassName))
      {
        auto l = resolve_fq(lookup.def);

        if (l.def)
        {
          Node inherit = l.def / Inherit;

          if ((inherit->type() != Inherit) || set.contains(inherit))
            return true;

          set.insert(inherit);
          worklist.emplace_back(set, inherit / Inherit);
        }
      }
      else if (
        (lookup.def->type() == FQType) &&
        ((lookup.def / Type)->type() == TypeAliasName))
      {
        auto l = resolve_fq(lookup.def);

        if (l.def)
          worklist.emplace_back(set, l);
      }
      else if (
        (lookup.def->type() == FQType) &&
        ((lookup.def / Type)->type() == TypeParamName))
      {
        auto l = resolve_fq(lookup.def);

        if (l.def)
        {
          auto find = lookup.bindings.find(l.def);

          if (find != lookup.bindings.end())
            worklist.emplace_back(set, lookup.make(find->second));
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

        In(Class) * T(Inherit)[Inherit] << T(Type) >> ([](Match& _) -> Node {
          if (recursive_inherit(_(Inherit)))
            return err(_[Inherit], "recursive inheritance");

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

// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../btype.h"
#include "../wf.h"

namespace verona
{
  bool recursive_typealias(Node node)
  {
    // This detects cycles in type aliases, which are not allowed. This happens
    // after type names are turned into FQType.
    assert(node == TypeAlias);

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

      if (lookup.def == Type)
      {
        worklist.emplace_back(set, lookup.make(lookup.def / Type));
      }
      else if (lookup.def->in({TypeTuple, TypeUnion, TypeIsect, TypeView}))
      {
        for (auto& t : *lookup.def)
          worklist.emplace_back(set, lookup.make(t));
      }
      else if ((lookup.def == FQType) && ((lookup.def / Type) == TypeAliasName))
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
      else if ((lookup.def == FQType) && ((lookup.def / Type) == TypeParamName))
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
    assert(node == Inherit);
    std::vector<std::pair<NodeSet, Lookup>> worklist;
    worklist.emplace_back(NodeSet{node}, node / Inherit);

    while (!worklist.empty())
    {
      auto work = worklist.back();
      auto& set = work.first;
      auto& lookup = work.second;
      worklist.pop_back();

      if (lookup.def == Type)
      {
        worklist.emplace_back(set, lookup.make(lookup.def / Type));
      }
      else if (lookup.def == TypeIsect)
      {
        for (auto& t : *lookup.def)
          worklist.emplace_back(set, lookup.make(t));
      }
      else if ((lookup.def == FQType) && ((lookup.def / Type) == TypeClassName))
      {
        auto l = resolve_fq(lookup.def);

        if (l.def)
        {
          Node inherit = l.def / Inherit;

          if ((inherit != Inherit) || set.contains(inherit))
            return true;

          set.insert(inherit);
          worklist.emplace_back(set, inherit / Inherit);
        }
      }
      else if ((lookup.def == FQType) && ((lookup.def / Type) == TypeAliasName))
      {
        auto l = resolve_fq(lookup.def);

        if (l.def)
          worklist.emplace_back(set, l);
      }
      else if ((lookup.def == FQType) && ((lookup.def / Type) == TypeParamName))
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

  bool valid_predicate(Node fq)
  {
    // A predicate is a type that can be used in a where clause. They can be
    // composed of unions and intersections of predicates and type aliases
    // that expand to predicates.
    Btype t = make_btype(fq);

    if (t != TypeAlias)
      return false;

    Btypes worklist;
    worklist.push_back(t);

    while (!worklist.empty())
    {
      t = worklist.back();
      worklist.pop_back();

      if (t == TypeSubtype)
      {
        // Do nothing.
      }
      else if (t->in({TypeUnion, TypeIsect}))
      {
        // Check that all children are valid predicates.
        std::for_each(t->node->begin(), t->node->end(), [&](auto& n) {
          worklist.push_back(t->make(n));
        });
      }
      else if (t == TypeAlias)
      {
        worklist.push_back(t / Type);
      }
      else
      {
        return false;
      }
    }

    return true;
  }

  bool valid_inherit(Node fq)
  {
    // A type that can be used in an inherit clause. They can be composed of
    // intersections of classes, traits, and type aliases that expand to
    // valid inherit clauses.
    Btype t = make_btype(fq);

    if (!t->in({Class, Trait, TypeAlias}))
      return false;

    Btypes worklist;
    worklist.push_back(t);

    while (!worklist.empty())
    {
      t = worklist.back();
      worklist.pop_back();

      if (t->in({Class, Trait}))
      {
        // Do nothing.
      }
      else if (t->in({Type, TypeIsect}))
      {
        // Check that all children are valid for code reuse.
        std::for_each(t->node->begin(), t->node->end(), [&](auto& n) {
          worklist.push_back(t->make(n));
        });
      }
      else if (t == TypeAlias)
      {
        worklist.push_back(t / Type);
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
      "typevalid",
      wfPassTypeFlat,
      dir::bottomup | dir::once,
      {
        T(TypeAlias)[TypeAlias] >> ([](Match& _) -> Node {
          if (recursive_typealias(_(TypeAlias)))
            return err(_(TypeAlias), "Recursive type alias");

          return NoChange;
        }),

        In(Class) * T(Inherit)[Inherit] << T(Type) >> ([](Match& _) -> Node {
          if (recursive_inherit(_(Inherit)))
            return err(_(Inherit), "Recursive inheritance");

          return NoChange;
        }),

        In(TypePred)++ * --(In(TypeSubtype, TypeArgs)++) * T(FQType)[FQType] >>
          ([](Match& _) -> Node {
            if (!valid_predicate(_(FQType)))
              return err(_(FQType), "This type isn't a valid type predicate");

            return NoChange;
          }),

        In(TypePred)++ * --(In(TypeSubtype, TypeArgs)++) *
            (TypeCaps /
             T(Trait, TypeTuple, Self, TypeList, TypeView, TypeVar, Package))
              [Type] >>
          [](Match& _) {
            return err(_(Type), "Can't put this in a type predicate");
          },

        In(Inherit)++ * --(In(TypeArgs)++) * T(FQType)[FQType] >>
          ([](Match& _) -> Node {
            if (!valid_inherit(_(FQType)))
              return err(_(FQType), "This type isn't valid for inheritance");

            return NoChange;
          }),

        In(Inherit)++ * --(In(TypeArgs)++) *
            (TypeCaps /
             T(TypeTuple,
               Self,
               TypeList,
               TypeView,
               TypeUnion,
               TypeVar,
               Package,
               TypeSubtype,
               TypeTrue,
               TypeFalse))[Type] >>
          [](Match& _) { return err(_(Type), "Can't inherit from this type"); },
      }};
  }
}

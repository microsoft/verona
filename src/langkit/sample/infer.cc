// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"
#include "wf.h"

namespace sample
{
  namespace detail
  {
    struct Bounds
    {
      // The lower bounds is a disjunction of assignments.
      std::set<Node, std::owner_less<>> lower;
      // The upper bounds is a conjunction of uses.
      std::set<Node, std::owner_less<>> upper;
    };

    using BoundsMap = std::map<Node, Bounds, std::owner_less<>>;
    using Cache = std::map<std::pair<Node, Node>, bool>;

    struct Checker
    {
      BoundsMap bounds;
      Cache cache;
      LookupDef<bool> match;

      Checker()
      {
        // TODO: view, func, isect, union, trait, refclass, reftypealias,
        // reftypeparam
        match.rules({
          T(TypeVar)[lhs] * T(TypeVar)[rhs] >>
            [this](auto& _) {
              auto r = _(rhs);
              auto& b = bounds[_(lhs)];
              b.upper.insert(r);
              bool ok = true;
              std::for_each(
                b.lower.begin(), b.lower.end(), [this, r, &ok](auto l) {
                  ok &= sub(l, r);
                });
              return ok;
            },

          T(TypeVar)[lhs] * Any[rhs] >>
            [this](auto& _) {
              bounds[_(lhs)].upper.insert(_(rhs));
              return true;
            },

          Any[lhs] * T(TypeVar)[rhs] >>
            [this](auto& _) {
              bounds[_(rhs)].lower.insert(_(lhs));
              return true;
            },

          T(TypeTuple)[lhs] * T(TypeTuple)[rhs] >>
            [this](auto& _) {
              return (_(lhs)->size() == _(rhs)->size()) &&
                std::inner_product(
                       _(lhs)->begin(),
                       _(lhs)->end(),
                       _(rhs)->begin(),
                       true,
                       [](auto l, auto r) { return l && r; },
                       [this](auto l, auto r) { return sub(l, r); });
            },

          (T(TypeThrow) << Any[lhs]) * (T(TypeThrow) << Any[rhs]) >>
            [this](auto& _) { return sub(_(lhs), _(rhs)); },

          T(RefClass)[lhs] * T(RefClass)[rhs] >>
            [this](auto& _) {
              // TODO: typeargs have to be the same
              auto l = look->at(_[lhs]);
              auto r = look->at(_[rhs]);
              return l.def == r.def;
            },
        });
      }

      bool sub(Node lhs, Node rhs)
      {
        // Don't repeat checks. Initially assume the check succeeds.
        auto [it, fresh] = cache.try_emplace({lhs, rhs}, true);
        return fresh ? match.at(lhs, rhs) : it->second;
      }
    };
  }

  Pass infer()
  {
    // TODO: when done, check all lower <: upper
    auto inferdef = std::make_shared<PassDef>();
    auto infer = inferdef.get();
    auto check = std::make_shared<detail::Checker>();

    // infer->rules({
    //   T(Lift)

    // });

    return inferdef;
  }
}

// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"
#include "wf.h"

namespace sample
{
  void typeargs(Found& found, Node ta)
  {
    // TODO: what if def is a Typeparam?
    // use the bounds somehow?
    if (!found.def || !ta)
      return;

    // TODO: error node if it's something that doesn't take typeargs?
    if (!found.def->type().in({Class, Function, Typealias}))
      return;

    auto tp = found.def->at(
      wf / Class / Typeparams,
      wf / Function / Typeparams,
      wf / Typealias / Typeparams);

    // TODO: error node if there are too many typeargs?
    Nodes args{ta->begin(), ta->end()};
    args.resize(tp->size());

    std::transform(
      tp->begin(),
      tp->end(),
      args.begin(),
      std::inserter(found.map, found.map.end()),
      [](auto param, auto arg) { return std::make_pair(param, arg); });
  }

  Node lookdown(Found& found, Node id)
  {
    NodeSet visited;

    while (true)
    {
      // Check if we've visited this node before. If so, we've found a cycle.
      auto [it, inserted] = visited.insert(found.def);
      if (!inserted)
        return {};

      if (found.def->type() == Class)
      {
        return found.def->lookdown_first(id);
      }
      else if (found.def->type() == Typeparam)
      {
        auto it = found.map.find(found.def);
        if ((it != found.map.end()) && it->second)
        {
          found.def = it->second;
          continue;
        }

        auto bounds = found.def->at(wf / Typeparam / Bounds);
        if (!bounds->empty())
        {
          found.def = bounds;
          continue;
        }
      }
      else if (found.def->type() == Typealias)
      {
        found.def = found.def->at(wf / Typealias / Type);
        continue;
      }
      else if (found.def->type() == Type)
      {
        found.def = found.def->at(wf / Type / Type);
        continue;
      }
      else if (found.def->type() == TypeView)
      {
        found.def = found.def->at(wf / TypeView / rhs);
        continue;
      }
      else if (found.def->type() == TypeThrow)
      {
        found.def = found.def->at(wf / TypeThrow / Type);
        continue;
      }
      else if (found.def->type() == RefType)
      {
        auto ident = found.def->at(wf / RefType / Ident);
        auto ta = found.def->at(wf / RefType / Typeargs);
        found.def = ident->lookup_first();
        typeargs(found, ta);
        continue;
      }
      else if (found.def->type() == TypeIsect)
      {
        // TODO:
      }
      // TODO: typeisect, typeunion

      // Other nodes don't have children to look down.
      break;
    }

    return {};
  }

  Lookup<Found> lookup()
  {
    auto lookdef = std::make_shared<LookupDef<Found>>();
    auto look = lookdef.get();

    look->rules({
      // Look through an outer Type node.
      (T(Type) << (Any[Type])) * End >>
        [=](Match& _) { return look->at(_(Type)); },

      // An identifier and optional typeargs.
      ((T(Ident)[id] * ~T(Typeargs)[Typeargs]) /
       (T(RefType) << (T(Ident)[id] * T(Typeargs)[Typeargs] * End))) *
          End >>
        [=](Match& _) {
          Found found(_(id)->lookup_first());
          typeargs(found, _(Typeargs));
          return std::move(found);
        },

      // Nested lookup.
      ((T(RefType)[lhs] * T(DoubleColon) * T(Ident)[id] *
        ~T(Typeargs)[Typeargs]) /
       (T(RefType) << (T(RefType)[lhs] * T(Ident)[id] * T(Typeargs)[Typeargs]) *
          End)) *
          End >>
        [=](Match& _) {
          auto found = look->at(_(lhs));
          found.def = lookdown(found, _(id));
          typeargs(found, _(Typeargs));
          return std::move(found);
        },
    });

    return lookdef;
  }
}

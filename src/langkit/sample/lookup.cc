// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"
#include "wf.h"

namespace sample
{
  Lookup<Found> lookup()
  {
    struct State
    {
      Found found;
      std::vector<Node> aliases;
      size_t depth;
    };

    auto state = std::make_shared<State>();
    auto lookdef = std::make_shared<LookupDef<Found>>();
    auto look = lookdef.get();

    look->pre([=]() { ++state->depth; });
    look->post([=]() {
      if ((--state->depth) == 0)
      {
        *state = {};
      }
    });

    auto ret = [=](Node def) {
      auto found = state->found;
      found.def = def;
      // std::erase_if(found.map, [&](auto& sub) {
      //   if (!sub.second)
      //     return true;
      //   auto find = def->lookup_all(sub.first->location());
      //   return std::find(find.begin(), find.end(), sub.first) == find.end();
      // });
      return found;
    };

    auto typeargs = [=](auto& _, Node def) {
      // TODO: what if def is a Typeparam?
      // use the bounds somehow
      auto ta = _(Typeargs);
      if (!def || !ta)
        return;

      constexpr std::array<Token, 3> list{Typealias, Class, Function};
      auto it = std::find(list.begin(), list.end(), def->type());
      if (it == list.end())
        return;

      auto tp = def->at(
        wf / Typealias / Typeparams,
        wf / Class / Typeparams,
        wf / Function / Typeparams);

      std::vector<Node> args{ta->begin(), ta->end()};
      args.resize(tp->size());
      auto& found = state->found;

      std::transform(
        tp->begin(),
        tp->end(),
        args.begin(),
        std::inserter(found.map, found.map.end()),
        [](auto param, auto arg) {
          return std::make_pair(param, arg);
        });
    };

    auto sub = [=](Node def) {
      if (!def || (def->type() != Typeparam))
        return def;

      auto& found = state->found;
      auto it = found.map.find(def);
      if ((it != found.map.end()) && it->second)
        return it->second;

      return def;
    };

    auto alias = [=](Node def) {
      auto& aliases = state->aliases;
      auto it = std::find(aliases.begin(), aliases.end(), def);
      if (it != aliases.end())
        return true;

      aliases.push_back(def);
      return false;
    };

    look->rules({
      T(Ident)[id] * ~T(Typeargs)[Typeargs] >>
        [=](auto& _) {
          auto def = _(id)->lookup_first();
          typeargs(_, def);
          return look->at(def);
        },

      (T(Var) / T(Let) / T(Param) / T(Class) / T(Function))[id] >>
        [=](auto& _) {
          return ret(_(id));
        },

      T(Type) << (Any[Type]) >> [=](auto& _) { return look->at(_(Type)); },

      T(Typealias)[id] >>
        [=](auto& _) {
          auto def = _(id);
          if (alias(def))
            return ret(def);
          return look->at(def->at(wf / Typealias / Default));
        },

      T(Typeparam)[id] >>
        [=](auto& _) {
          auto def = sub(_(id));
          if (def->type() != Typeparam)
            return look->at(def);

          auto bounds = def->at(wf / Typeparam / Bounds);
          if (bounds->empty())
            return ret(def);

          return look->at(bounds);
        },

      (T(RefClass) / T(RefTypealias) / T(RefTypeparam) / T(Package))[lhs] *
          T(DoubleColon) * T(Ident)[id] * ~T(Typeargs)[Typeargs] >>
        [=](auto& _) {
          auto l = look->at(_(lhs));
          auto def = l.def->lookdown_first(_(id));
          typeargs(_, def);
          return look->at(def);
        },

      (T(RefClass) / T(RefTypealias) / T(RefTypeparam) / T(Package))
          << (T(Ident) * T(Typeargs))[id] >>
        [=](auto& _) { return look->at(_[id]); },

      (T(RefClass) / T(RefTypealias) / T(RefTypeparam) / T(Package))
          << (Any[lhs] * T(Ident)[id] * T(Typeargs)[Typeargs]) >>
        [=](auto& _) {
          auto def = look->at(_(lhs)).def->lookdown_first(_(id));
          typeargs(_, def);
          return look->at(def);
        },
    });

    return lookdef;
  }
}

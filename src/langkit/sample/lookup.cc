// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"
#include "wf.h"

namespace sample
{
  Lookup<Node> lookup()
  {
    auto look = std::make_shared<LookupDef<Node>>();

    using Subs = std::map<Node, Node, std::owner_less<>>;
    using Aliases = std::vector<Node>;
    using State = std::vector<std::pair<Subs, Aliases>>;
    auto state = std::make_shared<State>();
    state->push_back({});

    look->post([state]() {
      assert(state->size() == 1);
      state->clear();
      state->push_back({});
    });

    auto typeargs = [&look, state](auto& _, Node def) {
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

      std::vector<Node> args;
      std::transform(
        ta->begin(), ta->end(), std::back_inserter(args), [=](auto& arg) {
          state->push_back({});
          auto def = look->at(arg);
          state->pop_back();
          return def;
        });

      args.resize(tp->size());
      auto& subs = state->back().first;

      std::transform(
        tp->begin(),
        tp->end(),
        args.begin(),
        std::inserter(subs, subs.end()),
        [](auto& param, auto& arg) { return std::make_pair(param, arg); });
    };

    auto sub = [state](Node def) {
      if (!def || (def->type() != Typeparam))
        return def;

      auto& subs = state->back().first;
      auto it = subs.find(def);
      if ((it != subs.end()) && it->second)
        return it->second;

      return def;
    };

    auto alias = [state](Node def) {
      auto& aliases = state->back().second;
      auto it = std::find(aliases.begin(), aliases.end(), def);

      if (it != aliases.end())
        return true;

      aliases.push_back(def);
      return false;
    };

    look->rules({
      T(Ident)[id] * ~T(Typeargs)[Typeargs] >>
        [&look, typeargs](auto& _) {
          auto def = _(id)->lookup_first();
          typeargs(_, def);
          return look->at(def);
        },

      (T(Var) / T(Let) / T(Param) / T(Class) / T(Function))[id] >>
        [](auto& _) { return _(id); },

      T(Type) << Any[Type] >> [=](auto& _) { return look->at(_(Type)); },

      T(Typealias)[id] >>
        [alias, &look](auto& _) {
          auto def = _(id);
          if (alias(def))
            return def;
          return look->at(def->at(wf / Typealias / Default));
        },

      T(Typeparam)[id] >>
        [&look, sub](auto& _) {
          auto def = sub(_(id));
          if (def->type() != Typeparam)
            return def;
          auto bounds = def->at(wf / Typeparam / Bounds);
          return bounds->empty() ? def : look->at(bounds);
        },

      (T(RefClass) / T(RefTypealias) / T(RefTypeparam) / T(Package))[lhs] *
          T(DoubleColon) * T(Ident)[id] * ~T(Typeargs)[Typeargs] >>
        [&look, typeargs](auto& _) {
          auto def = look->at(_(lhs))->lookdown_first(_(id));
          typeargs(_, def);
          return look->at(def);
        },

      (T(RefClass) / T(RefTypealias) / T(RefTypeparam) / T(Package))
          << (T(Ident) * T(Typeargs))[id] >>
        [&look](auto& _) { return look->at(_(id)); },

      (T(RefClass) / T(RefTypealias) / T(RefTypeparam) / T(Package))
          << (Any[lhs] * T(Ident)[id] * T(Typeargs)[Typeargs]) >>
        [typeargs, &look](auto& _) {
          auto def = look->at(_(lhs))->lookdown_first(_(id));
          typeargs(_, def);
          return look->at(def);
        },
    });

    return look;
  }
}

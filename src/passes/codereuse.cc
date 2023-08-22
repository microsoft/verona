// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../btype.h"

namespace verona
{
  PassDef codereuse()
  {
    return {
      dir::once | dir::topdown,
      {
        T(Class)[Class]
            << (T(Ident)[Ident] * T(TypeParams)[TypeParams] *
                T(Inherit)[Inherit] * T(TypePred)[TypePred] *
                T(ClassBody)[ClassBody]) >>
          [](Match& _) {
            auto cls = _(Class);
            auto from = _(Inherit) / Inherit;

            std::map<Location, Nodes> reuse;

            std::vector<Btype> worklist;
            worklist.emplace_back(make_btype(_(Inherit) / Inherit));

            while (!worklist.empty())
            {
              auto type = worklist.back();
              worklist.pop_back();

              if (type->type() == TypeIsect)
              {
                for (auto& t : *type->node)
                  worklist.emplace_back(type->make(t));
              }
              else if (type->type() == TypeAlias)
              {
                worklist.emplace_back(type->field(Type));
              }
              else if (type->type().in({Class, TypeTrait}))
              {
                // TODO:
                // Reuse stuff if (a) it's not ambiguous and (b) it's not
                // already provided in ClassBody. Need to do type substitution.
                auto body = type->node / ClassBody;

                for (auto node : *body)
                {
                  if (node->type().in({FieldLet, FieldVar, Function}))
                  {
                    auto id = node / Ident;
                    auto defs = cls->lookdown(id->location());

                    for (auto def : defs)
                    {
                      if (def->type().in({FieldLet, FieldVar, Function}))
                      {
                      }
                    }

                    reuse[id->location()].push_back(node);
                  }
                }
              }
            }

            return Class << _(Ident) << _(TypeParams) << _(TypePred)
                         << _(ClassBody);
          },
      }};
  }
}

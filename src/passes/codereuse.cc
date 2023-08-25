// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../btype.h"

namespace verona
{
  PassDef codereuse()
  {
    return {
      T(Class)[Class]
          << (T(Ident)[Ident] * T(TypeParams)[TypeParams] *
              (T(Inherit) << T(Type)[Inherit]) * T(TypePred)[TypePred] *
              T(ClassBody)[ClassBody]) >>
        ([](Match& _) -> Node {
          std::vector<Btype> worklist;
          std::vector<Btype> inherit;
          worklist.emplace_back(make_btype(_(Inherit)));

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
            else if (type->type() == TypeTrait)
            {
              inherit.push_back(type);
            }
            else if (type->type() == Class)
            {
              // A super-class needs to have done its own codereuse pass.
              // This class will be processed later.
              if ((type->node / Inherit / Inherit)->type() != DontCare)
                return NoChange;

              inherit.push_back(type);
            }
          }

          auto body = _(ClassBody);

          for (auto& from : inherit)
          {
            for (auto node : *(from->node / ClassBody))
            {
              if (!node->type().in({FieldLet, FieldVar, Function}))
                continue;

              // Don't inherit functions without implementations.
              if (
                (node->type() == Function) &&
                ((node / Block)->type() == DontCare))
                continue;

              // TODO: type substitution for type parameters.

              // Clone an implicit version into classbody.
              auto f = clone(node);
              (f / Implicit) = Implicit;
              body << f;
            }
          }

          return Class << _(Ident) << _(TypeParams) << (Inherit << DontCare)
                       << _(TypePred) << body;
        }),
    };
  }
}

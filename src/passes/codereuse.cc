// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../subtype.h"

namespace verona
{
  PassDef codereuse()
  {
    struct Pending
    {
      size_t rc = 0;
      Btypes inherit;
      Nodes pending;
    };

    auto pending = std::make_shared<NodeMap<Pending>>();
    auto ready = std::make_shared<Nodes>();

    PassDef codereuse = {
      dir::topdown | dir::once,
      {
        T(Class)[Class]
            << (T(Ident)[Ident] * T(TypeParams) *
                (T(Inherit) << T(Type)[Inherit]) * T(TypePred) *
                T(ClassBody)[ClassBody]) >>
          ([=](Match& _) -> Node {
            auto cls = _(Class);
            auto& pend = (*pending)[cls];
            Btypes worklist;
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
              else if (type->type() == Trait)
              {
                pend.inherit.push_back(type);
              }
              else if (type->type() == Class)
              {
                if ((type->node / Inherit / Inherit)->type() == Type)
                {
                  (*pending)[type->node].pending.push_back(cls);
                  pend.rc++;
                }

                pend.inherit.push_back(type);
              }
            }

            if (pend.rc == 0)
              ready->push_back(cls);

            return NoChange;
          }),
      }};

    codereuse.post([=](Node) {
      size_t changes = 0;

      while (!ready->empty())
      {
        auto cls = ready->back();
        ready->pop_back();
        auto& pend = (*pending)[cls];
        auto body = cls / ClassBody;
        (cls / Inherit) = Inherit << DontCare;

        for (auto& from : pend.inherit)
        {
          for (auto f : *(from->node / ClassBody))
          {
            if (!f->type().in({FieldLet, FieldVar, Function}))
              continue;

            // Don't inherit functions without implementations.
            if (
              (f->type() == Function) &&
              ((f / Block)->type() == DontCare))
              continue;

            // If we have an explicit version that conflicts, don't inherit.
            auto defs = cls->lookdown((f / Ident)->location());

            if (std::any_of(
                  defs.begin(), defs.end(), [&](Node& def) -> bool {
                    return conflict(f, def);
                  }))
              continue;

            // Clone an implicit version into classbody.
            f = clone(f);
            (f / Implicit) = Implicit;
            body << f;

            // TODO: type substitution for type parameters.
          }
        }

        for (auto& dep : pend.pending)
        {
          if (--pending->at(dep).rc == 0)
            ready->push_back(dep);
        }
      }

      pending->clear();
      return changes;
    });

    return codereuse;
  }
}

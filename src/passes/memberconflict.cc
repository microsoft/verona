// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"

namespace verona
{
  PassDef memberconflict()
  {
    return {
      dir::topdown | dir::once,
      {
        (T(FieldLet) / T(FieldVar) / T(Function))[Op] >> ([](Match& _) -> Node {
          auto f = _(Op);
          bool implicit = (f / Implicit)->type() == Implicit;
          auto defs = f->scope()->lookdown((f / Ident)->location());
          Nodes conflicts;

          for (auto& def : defs)
          {
            if (!conflict(f, def))
              continue;

            if (implicit == ((def / Implicit)->type() == Implicit))
            {
              // If both are implicit or both are explicit, it's an error.
              if (def->precedes(f))
              {
                // If they're identical, discard this one.
                if (def->equals(f))
                  return {};

                conflicts.push_back(def);
              }
            }
            else if (implicit)
            {
              // Discard the implicit definition.
              return {};
            }
          }

          if (!conflicts.empty())
          {
            Node e = Error;
            auto p = f->parent({Class, Trait});

            if (p->type() == Class)
              e << (ErrorMsg ^ "Member conflict in this class")
                << (ErrorAst ^ (p / Ident));
            else
              e << (ErrorMsg ^ "Member conflict in trait");

            e << (ErrorMsg ^ "This member conflicts")
              << ((ErrorAst ^ (f / Ident)) << f)
              << (ErrorMsg ^ "The conflicting member(s) are");

            for (auto& def : conflicts)
              e << (ErrorAst ^ (def / Ident));

            return e;
          }

          return NoChange;
        }),
      }};
  }
}

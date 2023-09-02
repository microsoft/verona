// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"

namespace verona
{
  PassDef localvar()
  {
    return {
      T(Var) << T(Ident)[Ident] >>
        [](Match& _) {
          return Assign << (Expr << (Let << _(Ident))) << (Expr << cell());
        },

      T(RefVar)[RefVar] >> [](Match& _) { return load(RefLet << *_[RefVar]); },

      T(RefVarLHS)[RefVarLHS] >>
        [](Match& _) { return RefLet << *_[RefVarLHS]; },
    };
  }
}
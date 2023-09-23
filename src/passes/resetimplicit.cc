// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../wf.h"

namespace verona
{
  PassDef resetimplicit()
  {
    return {
      "resetimplicit",
      wfPassResetImplicit,
      dir::bottomup | dir::once,
      {
        // Reset everything marked as implicit to be explicit.
        T(Implicit) >> ([](Match&) -> Node { return Explicit; }),

        // Strip implicit/explicit marker from fields.
        T(FieldLet) << (IsImplicit * T(Ident)[Ident] * T(Type)[Type]) >>
          [](Match& _) { return FieldLet << _(Ident) << _(Type); },

        T(FieldVar) << (IsImplicit * T(Ident)[Ident] * T(Type)[Type]) >>
          [](Match& _) { return FieldVar << _(Ident) << _(Type); },

        // Remove all `use` statements.
        In(ClassBody) * T(Use) >> ([](Match&) -> Node { return {}; }),

        In(Block) * T(Use) >> ([](Match&) -> Node { return Expr << Unit; }),
      }};
  }
}

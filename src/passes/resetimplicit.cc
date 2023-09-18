// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"

namespace verona
{
  PassDef resetimplicit()
  {
    return {
      dir::bottomup | dir::once,
      {
        // Strip implicit/explicit marker from fields.
        T(FieldLet, FieldVar)[FieldLet]
            << (IsImplicit * T(Ident)[Ident] * T(Type)[Type] * Any[Default]) >>
          [](Match& _) {
            Node field = _(FieldLet)->type();
            return field << _(Ident) << _(Type) << _(Default);
          },

        // Reset functions marked as implicit to be explicit.
        T(Function)[Function] << T(Implicit) >>
          [](Match& _) {
            auto f = _(Function);
            (f / Implicit) = Explicit;
            return f;
          },
      }};
  }
}

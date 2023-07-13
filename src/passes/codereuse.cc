// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"

namespace verona
{
  PassDef codereuse()
  {
    return {
      dir::once | dir::topdown,
      {
        T(Class)
            << (T(Ident)[Ident] * T(TypeParams)[TypeParams] *
                T(Inherit)[Inherit] * T(TypePred)[TypePred] *
                T(ClassBody)[ClassBody]) >>
          [](Match& _) {
            // TODO:
            // reuse stuff in Type if (a) it's not ambiguous and (b) it's not
            // already provided in ClassBody
            // need to do type substitution
            // strip Inherit from here on
            return Class << _(Ident) << _(TypeParams) << _(Inherit)
                         << _(TypePred) << _(ClassBody);
          },
      }};
  }
}

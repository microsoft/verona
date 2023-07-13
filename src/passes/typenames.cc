// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"

#include "lookup.h"

namespace verona
{
  PassDef typenames()
  {
    return {
      TypeStruct * T(DontCare)[DontCare] >>
        [](Match& _) { return TypeVar ^ _.fresh(l_typevar); },

      // Names on their own must be types.
      TypeStruct * T(Ident)[Id] * ~T(TypeArgs)[TypeArgs] >>
        [](Match& _) {
          return makename(DontCare, _(Id), (_(TypeArgs) || TypeArgs));
        },

      // Scoping binds most tightly.
      TypeStruct * TypeName[Lhs] * T(DoubleColon) * T(Ident)[Id] *
          ~T(TypeArgs)[TypeArgs] >>
        [](Match& _) {
          return makename(_(Lhs), _(Id), (_(TypeArgs) || TypeArgs));
        },
    };
  }
}

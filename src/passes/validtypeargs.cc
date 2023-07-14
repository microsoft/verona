// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../subtype.h"

namespace verona
{
  PassDef validtypeargs()
  {
    return {
      dir::bottomup | dir::once,
      {
        (TypeName[Op] << ((TypeName / T(DontCare)) * T(Ident) * T(TypeArgs))) >>
          ([](Match& _) -> Node {
            auto tn = _(Op);

            if (!is_implicit(tn) && !valid_typeargs(tn))
              return err(_[Op], "invalid type arguments");

            return NoChange;
          }),
      }};
  }
}

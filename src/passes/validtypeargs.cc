// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"
#include "subtype.h"

namespace verona
{
  PassDef validtypeargs()
  {
    return {
      dir::bottomup | dir::once,
      {
        (TypeName[Op] << ((TypeName / T(DontCare)) * T(Ident) * T(TypeArgs))) >>
          ([](Match& _) -> Node {
            if (!valid_typeargs(_(Op)))
              return err(_[Op], "invalid type arguments");

            return NoChange;
          }),
      }};
  }
}

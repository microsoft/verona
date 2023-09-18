// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../lookup.h"

namespace verona
{
  PassDef defbeforeuse()
  {
    return {
      dir::bottomup | dir::once,
      {
        T(RefLet) << T(Ident)[Ident] >> ([](Match& _) -> Node {
          if (!is_implicit(_(Ident)) && !lookup_type(_(Ident), {Bind, Param}))
            return err(_(Ident), "Use of uninitialized identifier");

          return NoChange;
        }),
      }};
  }
}

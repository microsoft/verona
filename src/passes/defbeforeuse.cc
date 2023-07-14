// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"

namespace verona
{
  PassDef defbeforeuse()
  {
    return {
      dir::topdown | dir::once,
      {
        T(RefLet) << T(Ident)[Ident] >> ([](Match& _) -> Node {
          auto id = _(Ident);
          auto defs = id->lookup();

          if (
            (defs.size() == 1) &&
            ((defs.front()->type() == Param) || defs.front()->precedes(id)))
            return NoChange;

          return err(_[Ident], "use of uninitialized identifier");
        }),
      }};
  }
}

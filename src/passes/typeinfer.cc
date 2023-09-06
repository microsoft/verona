// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../subtype.h"

namespace verona
{
  Node gamma(Node node)
  {
    assert(node->in({Move, Copy}));
    auto defs = (node / Ident)->lookup();
    assert(defs.size() == 1);
    auto def = defs.front();
    assert(def->in({Param, Bind}));
    return def / Type;
  }

  PassDef typeinfer()
  {
    return {
      dir::bottomup | dir::once,
      {
        T(Function)[Function] >> ([](Match& _) -> Node {
          auto f = _(Function);
          auto block = f / Block;

          for (auto stmt : *block)
          {
            if (stmt == Bind)
            {

            }
            else if (stmt == Drop)
            {

            }
          }

          return NoChange;
        }),
      }};
  }
}

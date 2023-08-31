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
        (T(FQType) / T(FQFunction))[Type] >> ([](Match& _) -> Node {
          auto tn = _(Type);

          if (is_implicit(tn))
            return NoChange;

          auto bt = make_btype(tn);

          // Ignore TypeParams and TypeTraits, as they don't have predicates.
          // If this fails to resolve to a definition, ignore it. It's either
          // test code, or the LHS has an error.
          if (!bt->type().in({Class, TypeAlias, Function}))
            return NoChange;

          if (!subtype(make_btype(TypeTrue), bt->field(TypePred)))
            return err(tn, "Invalid type arguments");

          return NoChange;
        }),
      }};
  }
}

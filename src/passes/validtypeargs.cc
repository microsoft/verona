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
        T(TypeParamName)[TypeParam] >> ([](Match& _) -> Node {
          auto tp = _(TypeParam);

          if (!(tp / TypeArgs)->empty())
            return err(tp, "type parameters can't have type arguments");

          return NoChange;
        }),

        (T(TypeClassName) / T(TypeAliasName) / T(FunctionName))[Type] >>
          ([](Match& _) -> Node {
            auto tn = _(Type);

            if (is_implicit(tn))
              return NoChange;

            auto bt = make_btype(tn);

            // If this fails to resolve to a definition, ignore it. It's either
            // test code, or the LHS has an error.
            if (!bt->type().in({Class, TypeAlias, Function}))
              return NoChange;

            if ((tn / TypeArgs)->size() > (bt->node / TypeParams)->size())
              return err(tn, "too many type arguments");

            // TODO: picking up the predicate for the predicate is wrong
            if (!subtype(make_btype(TypeTrue), bt->field(TypePred)))
              return err(tn, "invalid type arguments");

            return NoChange;
          }),
      }};
  }
}

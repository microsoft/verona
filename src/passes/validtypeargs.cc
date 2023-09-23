// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../subtype.h"
#include "../wf.h"

namespace verona
{
  PassDef validtypeargs()
  {
    auto preds = std::make_shared<Btypes>();

    PassDef pass = {
      "validtypeargs",
      wfPassDrop,
      dir::bottomup | dir::once,
      {
        T(Class, TypeAlias, Function) >> ([=](Match&) -> Node {
          preds->pop_back();
          return NoChange;
        }),

        T(FQType, FQFunction)[Type] >> ([=](Match& _) -> Node {
          auto tn = _(Type);

          if (is_implicit(tn))
            return NoChange;

          auto bt = make_btype(tn);

          // Ignore TypeParams and TypeTraits, as they don't have predicates.
          // If this fails to resolve to a definition, ignore it. It's either
          // test code, or the LHS has an error.
          if (!bt->in({Class, TypeAlias, Function}))
            return NoChange;

          if (!subtype(*preds, make_btype(TypeTrue), bt / TypePred))
            return err(tn, "Invalid type arguments");

          return NoChange;
        }),
      }};

    pass.pre(Class, [=](Node n) {
      preds->push_back(make_btype(n / TypePred));
      return 0;
    });

    pass.pre(TypeAlias, [=](Node n) {
      preds->push_back(make_btype(n / TypePred));
      return 0;
    });

    pass.pre(Function, [=](Node n) {
      preds->push_back(make_btype(n / TypePred));
      return 0;
    });

    return pass;
  }
}

// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../subtype.h"
#include "../wf.h"

namespace verona
{
  PassDef validtypeargs()
  {
    auto assume = std::make_shared<Btypes>();

    PassDef pass = {
      "validtypeargs",
      wfPassDrop,
      dir::bottomup | dir::once,
      {
        !In(FQFunction) * T(FQType)[Type] >> ([=](Match& _) -> Node {
          auto tn = _(Type);

          if (is_implicit(tn))
            return NoChange;

          auto bt = make_btype(tn);

          // Ignore TypeParams and TypeTraits, as they don't have predicates.
          // If this fails to resolve to a definition, ignore it. It's either
          // test code, or the LHS has an error.
          if (!bt->in({Class, TypeAlias, Function}))
            return NoChange;

          auto preds = all_predicates(bt->node);
          Nodes errs;

          for (auto& pred : preds)
          {
            if (!subtype(*assume, bt->make(pred)))
              errs.push_back(clone(pred));
          }

          if (errs.empty())
            return NoChange;

          auto e = err(tn, "Invalid type arguments");

          for (auto& err : errs)
            e << (ErrorMsg ^ "this predicate isn't satisfied") << err;

          return e;
        }),
      }};

    pass.pre({Class, TypeAlias, Function}, [=](Node n) {
      assume->push_back(make_btype(n / TypePred));
      return 0;
    });

    pass.post({Class, TypeAlias, Function}, [=](Node) {
      assume->pop_back();
      return 0;
    });

    return pass;
  }
}

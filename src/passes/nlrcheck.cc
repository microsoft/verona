// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"

namespace verona
{
  PassDef nlrcheck()
  {
    return {
      dir::topdown | dir::once,
      {
        T(NLRCheck)
            << (IsImplicit[Implicit] *
                (T(Call)[Call]
                 << ((T(Selector) / T(FQFunction))[Op] * T(Args)))) >>
          [](Match& _) {
            auto call = _(Call);

            if (is_llvm_call(_(Op)))
              return call;

            // Check the call result to see if it's a non-local return. If it
            // is, optionally unwrap it and return. Otherwise, continue
            // execution.
            auto id = _.fresh();
            auto ref = Expr << (RefLet << (Ident ^ id));
            auto nlr = Type << nonlocal(_);
            Node ret = Cast << ref << nlr;

            // Unwrap if we're in a function (Explicit), but not if we're in a
            // lambda (Implicit).
            if (_(Implicit) == Explicit)
              ret = load(ret, true);

            return ExprSeq << (Expr
                               << (Bind << (Ident ^ id) << typevar(_)
                                        << (Expr << call)))
                           << (Expr
                               << (Conditional
                                   << (Expr
                                       << (TypeTest << clone(ref)
                                                    << clone(nlr)))
                                   << (Block << (Return << (Expr << ret)))
                                   << (Block << clone(ref))));
          },
      }};
  }
}

// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../wf.h"

namespace verona
{
  PassDef nlrcheck()
  {
    return {
      "nlrcheck",
      wfPassNLRCheck,
      dir::bottomup | dir::once,
      {
        T(NLRCheck)
            << (T(Call)[Call] << (T(Selector, FQFunction)[Op] * T(Args))) >>
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

            // Unwrap if we're in a function, but not if we're in a lambda.
            // Remove the NLRCheck around the load call.
            if (call->parent(Function) / Implicit != LambdaFunc)
              ret = load(ret) / Call;

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

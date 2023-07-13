// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"

namespace verona
{
  PassDef nlrcheck()
  {
    return {
      dir::topdown | dir::once,
      {
        T(NLRCheck) << ((T(Call) / T(CallLHS))[Call]) >>
          [](Match& _) {
            auto call = _(Call);
            return nlrexpand(
              _, call, call->parent({Lambda, Function})->type() == Function);
          },
      }};
  }
}

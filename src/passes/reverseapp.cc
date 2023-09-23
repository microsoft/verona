// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../wf.h"

namespace verona
{
  PassDef reverseapp()
  {
    return {
      "reverseapp",
      wfPassReverseApp,
      dir::topdown,
      {
        // Dot: reverse application. This binds most strongly.
        (Object / Operator)[Lhs] * T(Dot) * Operator[Rhs] >>
          [](Match& _) { return call(_(Rhs), _(Lhs)); },

        (Object / Operator)[Lhs] * T(Dot) * Object[Rhs] >>
          [](Match& _) { return call(selector(l_apply), _(Rhs), _(Lhs)); },

        T(Dot)[Dot] >>
          [](Match& _) {
            return err(_(Dot), "Must use `.` with values and operators");
          },
      }};
  }
}

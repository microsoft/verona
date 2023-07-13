// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"

namespace verona
{
  PassDef reverseapp()
  {
    return {
      // Dot: reverse application. This binds most strongly.
      (Object / Operator)[Lhs] * T(Dot) * Operator[Rhs] >>
        [](Match& _) { return call(_(Rhs), _(Lhs)); },

      (Object / Operator)[Lhs] * T(Dot) * Object[Rhs] >>
        [](Match& _) { return call(apply(), _(Rhs), _(Lhs)); },

      T(Dot)[Dot] >>
        [](Match& _) {
          return err(_[Dot], "must use `.` with values and operators");
        },
    };
  }
}

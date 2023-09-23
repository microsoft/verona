// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../wf.h"

namespace verona
{
  PassDef typeview()
  {
    return {
      "typeview",
      wfPassTypeView,
      dir::topdown,
      {
        // Viewpoint adaptation binds more tightly than function types.
        TypeStruct * TypeElem[Lhs] * T(Dot) * TypeElem[Rhs] >>
          [](Match& _) {
            return TypeView << (Type << _[Lhs]) << (Type << _[Rhs]);
          },

        // TypeList binds more tightly than function types.
        TypeStruct * TypeElem[Lhs] * T(Ellipsis) >>
          [](Match& _) { return TypeList << (Type << _[Lhs]); },

        TypeStruct * T(DoubleColon)[DoubleColon] >>
          [](Match& _) { return err(_(DoubleColon), "Misplaced type scope"); },
        TypeStruct * T(TypeArgs)[TypeArgs] >>
          [](Match& _) {
            return err(
              _(TypeArgs), "Type arguments on their own are not a type");
          },
        TypeStruct * T(Dot)[Dot] >>
          [](Match& _) { return err(_(Dot), "Misplaced type viewpoint"); },
        TypeStruct * T(Ellipsis)[Ellipsis] >>
          [](Match& _) { return err(_(Ellipsis), "Misplaced type list"); },
      }};
  }
}

// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../wf.h"

namespace verona
{
  PassDef typealg()
  {
    return {
      "typealg",
      wfPassTypeAlg,
      dir::topdown,
      {
        // Build algebraic types.
        TypeStruct * TypeElem[Lhs] * T(Symbol, "&") * TypeElem[Rhs] >>
          [](Match& _) {
            return TypeIsect << (Type << _[Lhs]) << (Type << _[Rhs]);
          },
        TypeStruct * TypeElem[Lhs] * T(Symbol, "\\|") * TypeElem[Rhs] >>
          [](Match& _) {
            return TypeUnion << (Type << _[Lhs]) << (Type << _[Rhs]);
          },
        TypeStruct * TypeElem[Lhs] * T(Symbol, "<") * TypeElem[Rhs] >>
          [](Match& _) {
            return TypeSubtype << (Type << _[Lhs]) << (Type << _[Rhs]);
          },

        TypeStruct * T(Symbol)[Symbol] >>
          [](Match& _) { return err(_(Symbol), "Invalid symbol in type"); },
      }};
  }
}

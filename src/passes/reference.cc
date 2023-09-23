// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../lookup.h"
#include "../wf.h"

namespace verona
{
  PassDef reference()
  {
    return {
      "reference",
      wfPassReference,
      dir::topdown,
      {
        // LLVM literal.
        In(Expr) * T(LLVM)[LLVM] * T(Ident)[Lhs] * T(Ident)++[Rhs] >>
          [](Match& _) {
            auto rhs = _[Rhs];
            auto s = std::string()
                       .append(_(LLVM)->location().view())
                       .append(" %")
                       .append(_(Lhs)->location().view());

            for (auto& i = rhs.first; i != rhs.second; ++i)
              s.append(", %").append((*i)->location().view());

            return LLVM ^ s;
          },

        In(Expr) * T(LLVM)[Lhs] * T(LLVM)[Rhs] >>
          [](Match& _) {
            return LLVM ^
              std::string()
                .append(_(Lhs)->location().view())
                .append(" ")
                .append(_(Rhs)->location().view());
          },

        // Dot and DoubleColon notation. Use `Ident` as a selector, even if it's
        // in scope.
        In(Expr) * T(Dot, DoubleColon)[Dot] * T(Ident, Symbol)[Ident] *
            ~T(TypeArgs)[TypeArgs] >>
          [](Match& _) {
            return Seq << _(Dot) << selector(_(Ident), _(TypeArgs));
          },

        // Local reference.
        In(Expr) * T(Ident)[Ident] >> ([](Match& _) -> Node {
          auto id = _(Ident);

          if (lookup_type(id, {Var}))
            return RefVar << id;
          else if (lookup_type(id, {Let, Param}))
            return RefLet << id;

          return NoChange;
        }),

        // Given `try { ... }`, apply the lambda.
        In(Expr) * T(Try)[Try] * T(Lambda)[Lambda] * --T(Dot) >>
          [](Match& _) {
            return Seq << Try << _(Lambda) << Dot << selector(l_apply);
          },
      }};
  }
}

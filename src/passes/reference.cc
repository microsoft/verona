// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../lookup.h"

namespace verona
{
  PassDef reference()
  {
    return {
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

      // Error out on invalid scoped references.
      In(Expr) * (T(DoubleColon) * !T(Selector))[DoubleColon] >>
        [](Match& _) {
          return err(_(DoubleColon), "Expected a scoped reference");
        },

      // Local reference.
      In(Expr) *
          T(Ident)[Ident]([](auto& n) { return lookup_type(n, {Var}); }) >>
        [](Match& _) { return RefVar << _(Ident); },

      In(Expr) * T(Ident)[Ident]([](auto& n) {
        return lookup_type(n, {Let, Param});
      }) >>
        [](Match& _) { return RefLet << _(Ident); },
    };
  }
}

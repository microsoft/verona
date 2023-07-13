// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"

namespace verona
{
  inline const auto Liftable = T(Tuple) / T(Call) / T(CallLHS) /
    T(Conditional) / T(FieldRef) / T(TypeTest) / T(Cast) / T(Selector) /
    T(FunctionName) / Literal;

  PassDef anf()
  {
    return {
      // This liftable expr is already bound from `let x = e`.
      In(Bind) * (T(Expr) << Liftable[Lift]) >>
        [](Match& _) { return _(Lift); },

      // Lift `let x` bindings, leaving a RefLet behind.
      T(Expr) << (T(Bind)[Bind] << (T(Ident)[Id] * T(Type) * T(Expr))) >>
        [](Match& _) {
          return Seq << (Lift << Block << _(Bind))
                     << (RefLet << (Ident ^ _(Id)));
        },

      // Lift RefLet and Return.
      T(Expr) << (T(RefLet) / T(Return))[Op] >> [](Match& _) { return _(Op); },

      // Lift LLVM literals that are at the block level.
      In(Block) * (T(Expr) << T(LLVM)[LLVM]) >>
        [](Match& _) { return _(LLVM); },

      // Create a new binding for this liftable expr.
      T(Expr)
          << (Liftable[Lift] /
              (T(TypeAssert)
               << ((Liftable / T(RefLet))[Lift] * T(Type)[Type]))) >>
        [](Match& _) {
          auto id = _.fresh();
          return Seq << (Lift << Block
                              << (Bind << (Ident ^ id) << typevar(_, Type)
                                       << _(Lift)))
                     << (RefLet << (Ident ^ id));
        },

      // Compact an ExprSeq with only one element.
      T(ExprSeq) << (Any[Lhs] * End) >> [](Match& _) { return _(Lhs); },

      // Discard leading RefLets in ExprSeq.
      In(ExprSeq) * T(RefLet) * Any[Lhs] >> [](Match& _) { return _(Lhs); },
    };
  }
}

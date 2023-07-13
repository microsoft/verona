// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"

namespace verona
{
  PassDef application()
  {
    // These rules allow expressions such as `-3 * -4` or `not a and not b` to
    // have the expected meaning.
    return {
      // Ref expressions.
      T(Ref) * T(RefVar)[RefVar] >>
        [](Match& _) { return RefVarLHS << *_[RefVar]; },

      T(Ref) * (T(NLRCheck) << T(Call)[Call]) >>
        [](Match& _) { return NLRCheck << (CallLHS << *_[Call]); },

      // Try expressions.
      T(Try) * (T(NLRCheck) << (T(Call) / T(CallLHS))[Call]) >>
        [](Match& _) { return _(Call); },

      T(Try) * T(Lambda)[Lambda] >>
        [](Match& _) {
          return Call << apply() << (Args << (Expr << _(Lambda)));
        },

      // Adjacency: application.
      In(Expr) * Object[Lhs] * Object[Rhs] >>
        [](Match& _) { return call(apply(), _(Lhs), _(Rhs)); },

      // Prefix. This doesn't rewrite `Op Op`.
      In(Expr) * Operator[Op] * Object[Rhs] >>
        [](Match& _) { return call(_(Op), _(Rhs)); },

      // Infix. This doesn't rewrite with an operator on Lhs or Rhs.
      In(Expr) * Object[Lhs] * Operator[Op] * Object[Rhs] >>
        [](Match& _) { return call(_(Op), _(Lhs), _(Rhs)); },

      // Zero argument call.
      In(Expr) * Operator[Op] * --(Object / Operator) >>
        [](Match& _) { return call(_(Op)); },

      // Tuple flattening.
      In(Tuple) * T(Expr) << (Object[Lhs] * T(Ellipsis) * End) >>
        [](Match& _) { return TupleFlatten << (Expr << _(Lhs)); },

      // Use `_` (DontCare) for partial application of arbitrary arguments.
      T(Call)
          << (Operator[Op] *
              (T(Args)
               << ((T(Expr) << !T(DontCare))++ *
                   (T(Expr)
                    << (T(DontCare) /
                        (T(TypeAssert) << (T(DontCare) * T(Type))))) *
                   T(Expr)++))[Args]) >>
        [](Match& _) {
          Node params = Params;
          Node args = Args;
          auto lambda = Lambda << TypeParams << params << typevar(_)
                               << typepred()
                               << (Block << (Expr << (Call << _(Op) << args)));

          for (auto& arg : *_(Args))
          {
            auto expr = arg->front();

            if (expr->type() == DontCare)
            {
              auto id = _.fresh(l_param);
              params << (Param << (Ident ^ id) << typevar(_) << DontCare);
              args << (Expr << (RefLet << (Ident ^ id)));
            }
            else if (expr->type() == TypeAssert)
            {
              auto id = _.fresh(l_param);
              params << (Param << (Ident ^ id) << (expr / Type) << DontCare);
              args << (Expr << (RefLet << (Ident ^ id)));
            }
            else
            {
              args << arg;
            }
          }

          return lambda;
        },

      // Remove the NLRCheck from a partial application.
      T(NLRCheck) << (T(Lambda)[Lambda] * End) >>
        [](Match& _) { return _(Lambda); },

      In(Expr) * T(DontCare) >>
        [](Match& _) {
          // Remaining DontCare are discarded bindings.
          return Let << (Ident ^ _.fresh());
        },

      // Turn remaining uses of Unit into std::builtin::Unit::create()
      T(Unit) >> [](Match&) { return unit(); },

      T(Ellipsis) >>
        [](Match& _) {
          return err(_[Ellipsis], "must use `...` after a value in a tuple");
        },

      // Compact expressions.
      In(Expr) * T(Expr) << (Any[Expr] * End) >>
        [](Match& _) { return _(Expr); },
      T(Expr) << (T(Expr)[Expr] * End) >> [](Match& _) { return _(Expr); },
    };
  }
}

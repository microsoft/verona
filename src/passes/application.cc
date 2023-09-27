// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../lookup.h"
#include "../wf.h"

namespace verona
{
  PassDef application()
  {
    // These rules allow expressions such as `-3 * -4` or `not a and not b` to
    // have the expected meaning.
    return {
      "application",
      wfPassApplication,
      dir::topdown,
      {
        // Ref expressions.
        T(Ref) * T(RefVar)[RefVar] >>
          [](Match& _) { return RefVarLHS << *_[RefVar]; },

        T(Ref) * (T(NLRCheck) << RhsCall[Call]) >>
          [](Match& _) { return NLRCheck << call_lhs(_(Call)); },

        // Try expressions.
        T(Try) * (T(NLRCheck) << T(Call)[Call]) >>
          [](Match& _) { return _(Call); },

        // Adjacency: application.
        In(Expr) * Object[Lhs] * Object[Rhs] >>
          [](Match& _) { return call(selector(l_apply), _(Lhs), _(Rhs)); },

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
        T(Call)[Call]
            << (Operator[Op] *
                (T(Args)
                 << ((T(Expr) << !T(DontCare))++ *
                     (T(Expr)
                      << (T(DontCare) /
                          (T(TypeAssert) << (T(DontCare) * T(Type))))) *
                     T(Expr)++))[Args]) >>
          [](Match& _) {
            // Create the anonymous type name.
            bool in_nlrcheck = _(Call)->parent() == NLRCheck;
            auto class_id = _.fresh(l_lambda);

            // Build an FQType for the anonymous type.
            auto fq = append_fq(
              local_fq(_(Call)->parent(Function)),
              TypeClassName << (Ident ^ class_id) << TypeArgs);

            // Start with a Self parameter.
            Node params = Params
              << (Param << (Ident ^ _.fresh(l_self)) << (Type << Self));
            Node args = Tuple;

            for (auto& arg : *_(Args))
            {
              auto expr = arg->front();

              if (expr == DontCare)
              {
                auto id = _.fresh(l_param);
                params << (Param << (Ident ^ id) << typevar(_));
                args << (Expr << (RefLet << (Ident ^ id)));
              }
              else if (expr == TypeAssert)
              {
                auto id = _.fresh(l_param);
                params << (Param << (Ident ^ id) << (expr / Type));
                args << (Expr << (RefLet << (Ident ^ id)));
              }
              else
              {
                args << arg;
              }
            }

            // Add the create and apply functions to the anonymous type.
            auto create_func = Function
              << Implicit << Rhs << (Ident ^ l_create) << TypeParams << Params
              << typevar(_) << DontCare << typepred()
              << (Block << (Expr << call(append_fq(fq, selector(l_new)))));

            auto apply_func = Function
              << LambdaFunc << Rhs << (Ident ^ l_apply) << TypeParams << params
              << typevar(_) << DontCare << typepred()
              << (Block << (Expr << call(_(Op), args)));

            auto classdef = Class << (Ident ^ class_id) << TypeParams
                                  << (Inherit << DontCare) << typepred()
                                  << (ClassBody << create_func << apply_func);

            auto create = call(append_fq(fq, selector(l_create)));

            if (in_nlrcheck)
              create = create / Call;

            return Seq << (Lift << Block << classdef) << create;
          },

        // Remaining DontCare are discarded bindings.
        In(Expr) * T(DontCare) >>
          [](Match& _) { return Let << (Ident ^ _.fresh()); },

        // Turn Unit into std::builtin::Unit::create()
        T(Unit) >> [](Match&) { return unit(); },

        // Turn True into std::builtin::Bool::make_true()
        T(True) >> [](Match&) { return booltrue(); },

        // Turn False into std::builtin::Bool::make_false()
        T(False) >> [](Match&) { return boolfalse(); },

        T(Ellipsis) >>
          [](Match& _) {
            return err(_(Ellipsis), "`...` must be after a value in a tuple");
          },

        // Compact expressions.
        In(Expr) * T(Expr) << (Any[Expr] * End) >>
          [](Match& _) { return _(Expr); },

        T(Expr) << (T(Expr)[Expr] * End) >> [](Match& _) { return _(Expr); },
      }};
  }
}

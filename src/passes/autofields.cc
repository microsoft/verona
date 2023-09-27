// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../wf.h"

namespace verona
{
  PassDef autofields()
  {
    return {
      "autofields",
      wfPassAutoFields,
      dir::bottomup | dir::once,
      {
        T(FieldVar, FieldLet)[Op] << (T(Ident)[Ident] * T(Type)[Type]) >>
          ([](Match& _) -> Node {
            // If it's a FieldLet, generate only an RHS function. If it's a
            // FieldVar, generate an LHS function, which will autogenerate an
            // RHS function.
            auto field = _(Op);
            auto id = _(Ident);
            auto self_id = _.fresh(l_self);
            Token hand = (field == FieldVar) ? Lhs : Rhs;
            auto expr = FieldRef << (RefLet << (Ident ^ self_id)) << clone(id);

            if (hand == Rhs)
              expr = load(expr);

            auto f = Function
              << Implicit << hand << clone(id) << TypeParams
              << (Params << (Param << (Ident ^ self_id) << (Type << Self)))
              << clone(_(Type)) << DontCare << typepred()
              << (Block << (Expr << expr));

            return Seq << field << f;
          }),
      }};
  }
}

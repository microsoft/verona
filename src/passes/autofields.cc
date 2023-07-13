// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"

namespace verona
{
  PassDef autofields()
  {
    return {
      dir::topdown | dir::once,
      {
        (T(FieldVar) / T(FieldLet))[Op] << (T(Ident)[Id] * T(Type)[Type]) >>
          ([](Match& _) -> Node {
            // If it's a FieldLet, generate only an RHS function. If it's a
            // FieldVar, generate an LHS function, which will autogenerate an
            // RHS function.
            auto field = _(Op);
            auto id = _(Id);
            auto self_id = _.fresh(l_self);
            Token is_ref = (field->type() == FieldVar) ? Ref : DontCare;
            auto expr = FieldRef << (RefLet << (Ident ^ self_id)) << clone(id);

            if (is_ref == DontCare)
              expr = load(expr);

            // TODO: capability for Self, return type is self.T
            auto f = Function << is_ref << clone(id) << TypeParams
                              << (Params
                                  << (Param << (Ident ^ self_id)
                                            << (Type << Self) << DontCare))
                              << clone(_(Type)) << DontCare << typepred()
                              << (Block << (Expr << expr));

            return Seq << field << f;
          }),
      }};
  }
}

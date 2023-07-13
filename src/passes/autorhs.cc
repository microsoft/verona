// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"

namespace verona
{
  PassDef autorhs()
  {
    return {
      dir::topdown | dir::once,
      {
        T(Function)[Function]
            << (T(Ref) * Name[Id] * T(TypeParams)[TypeParams] *
                T(Params)[Params] * T(Type)[Type] * T(DontCare) *
                T(TypePred)[TypePred] * (T(Block) / T(DontCare))) >>
          ([](Match& _) -> Node {
            auto f = _(Function);
            auto id = _(Id);
            auto params = _(Params);
            auto parent = f->parent()->parent()->shared_from_this();
            Token ptype =
              (parent->type() == Class) ? TypeClassName : TypeTraitName;
            auto tn = parent / Ident;
            auto defs = parent->lookdown(id->location());
            auto found = false;

            // Check if there's an RHS function with the same name and arity.
            for (auto def : defs)
            {
              if (
                (def != f) && (def->type() == Function) &&
                ((def / Ref)->type() != Ref) &&
                ((def / Ident)->location() == id->location()) &&
                ((def / Params)->size() == params->size()))
              {
                found = true;
                break;
              }
            }

            if (found)
              return NoChange;

            // If not, create an RHS function with the same name and arity.
            Node args = Args;

            for (auto param : *params)
              args << (Expr << (RefLet << clone(param / Ident)));

            auto rhs_f =
              Function << DontCare << clone(id) << clone(_(TypeParams))
                       << clone(params) << clone(_(Type)) << DontCare
                       << clone(_(TypePred))
                       << (Block
                           << (Expr << load(
                                 CallLHS << (FunctionName
                                             << (ptype << DontCare << clone(tn)
                                                       << TypeArgs)
                                             << clone(id) << TypeArgs)
                                         << args)));

            return Seq << f << rhs_f;
          }),
      }};
  }
}

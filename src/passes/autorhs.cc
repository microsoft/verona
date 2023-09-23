// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../lookup.h"
#include "../wf.h"

namespace verona
{
  PassDef autorhs()
  {
    return {
      "autorhs",
      wfPassAutoFields,
      dir::bottomup | dir::once,
      {
        T(Function)[Function]
            << (IsImplicit * T(Lhs) * T(Ident)[Ident] *
                T(TypeParams)[TypeParams] * T(Params)[Params] * T(Type)[Type] *
                T(DontCare) * T(TypePred)[TypePred] * T(Block, DontCare)) >>
          ([](Match& _) -> Node {
            auto f = _(Function);
            auto id = _(Ident);
            auto params = _(Params);
            auto parent = f->parent()->parent()->shared_from_this();
            auto tn = parent / Ident;
            auto defs = parent->lookdown(id->location());
            auto found = false;

            // Check if there's an RHS function with the same name and arity.
            for (auto def : defs)
            {
              if (
                (def != f) && (def == Function) && ((def / Ref) != Rhs) &&
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
            Node args = Tuple;

            for (auto param : *params)
              args << (Expr << (RefLet << clone(param / Ident)));

            // Call the LHS function with all the same type arguments, load the
            // result, and return that.
            auto rhs_f = Function
              << Implicit << Rhs << clone(id) << clone(_(TypeParams))
              << clone(params) << clone(_(Type)) << DontCare
              << clone(_(TypePred))
              << (Block << (Expr << load(call(local_fq(f), args))));

            return Seq << f << rhs_f;
          }),
      }};
  }
}

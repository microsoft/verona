// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../lookup.h"

namespace verona
{
  PassDef defaultargs()
  {
    return {
      dir::topdown | dir::once,
      {
        T(Function)[Function]
            << (T(Explicit) * Hand[Ref] * T(Ident)[Ident] *
                T(TypeParams)[TypeParams] *
                (T(Params)
                 << ((T(Param) << (T(Ident) * T(Type) * T(DontCare)))++[Lhs] *
                     (T(Param) << (T(Ident) * T(Type) * T(NLRCheck)))++[Rhs] *
                     End)) *
                T(Type)[Type] * T(DontCare) * T(TypePred)[TypePred] *
                (T(Block) / T(DontCare))[Block]) >>
          [](Match& _) {
            Node seq = Seq;
            auto hand = _(Ref);
            auto id = _(Ident);
            auto tp = _(TypeParams);
            auto ty = _(Type);
            auto pred = _(TypePred);
            auto lhs = _[Lhs];
            auto rhs = _[Rhs];

            auto fq = local_fq(_(Function));
            Node params = Params;
            Node args = Tuple;

            // Start with parameters that have no default value.
            for (auto it = lhs.first; it != lhs.second; ++it)
            {
              auto param_id = *it / Ident;
              params << (Param << clone(param_id) << clone(*it / Type));
              args << (Expr << (RefLet << clone(param_id)));
            }

            for (auto it = rhs.first; it != rhs.second; ++it)
            {
              // At this point, the default argument is a create call on the
              // anonymous class derived from the lambda. Apply the created
              // lambda to get the default argument.
              auto def_arg = call(selector(l_apply), (*it / Default));

              // Add the default argument to the forwarding call.
              args << (Expr << def_arg);

              // Add a new function that calls the arity+1 function. Mark it as
              // explicit, so that errors when type checking the default
              // arguments are reported.
              seq
                << (Function
                    << Explicit << clone(hand) << clone(id) << clone(tp)
                    << clone(params) << clone(ty) << DontCare << clone(pred)
                    << (Block << (Expr << (call(clone(fq), clone(args))))));

              // Add a parameter.
              auto param_id = *it / Ident;
              params << (Param << clone(param_id) << clone(*it / Type));

              // Replace the last argument with a reference to the parameter.
              args->pop_back();
              args << (Expr << (RefLet << clone(param_id)));
            }

            // The original function, with no default arguments.
            return seq
              << (Function << Explicit << hand << id << tp << params << ty
                           << DontCare << pred << _(Block));
          },

        T(Param) << (T(Ident)[Ident] * T(Type)[Type] * T(DontCare)) >>
          [](Match& _) { return Param << _(Ident) << _(Type); },

        T(Param)[Param] << (T(Ident) * T(Type) * T(NLRCheck)) >>
          [](Match& _) {
            return err(
              _[Param],
              "can't put a default value before a non-defaulted value");
          },
      }};
  }
}

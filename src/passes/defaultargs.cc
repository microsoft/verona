// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"

namespace verona
{
  PassDef defaultargs()
  {
    return {
      dir::topdown | dir::once,
      {
        T(Function)[Function]
            << ((T(Ref) / T(DontCare))[Ref] * Name[Id] *
                T(TypeParams)[TypeParams] *
                (T(Params)
                 << ((T(Param) << (T(Ident) * T(Type) * T(DontCare)))++[Lhs] *
                     (T(Param) << (T(Ident) * T(Type) * T(Call)))++[Rhs] *
                     End)) *
                T(Type)[Type] * T(DontCare) * T(TypePred)[TypePred] *
                (T(Block) / T(DontCare))[Block]) >>
          [](Match& _) {
            Node seq = Seq;
            auto ref = _(Ref);
            auto id = _(Id);
            auto tp = _(TypeParams);
            auto ty = _(Type);
            auto pred = _(TypePred);
            Node params = Params;
            Node call = (ref->type() == Ref) ? CallLHS : Call;

            auto parent = _(Function)->parent()->parent()->shared_from_this();
            auto tn = parent / Ident;
            Token ptype =
              (parent->type() == Class) ? TypeClassName : TypeTraitName;
            Node args = Args;
            auto fwd = Expr
              << (call << (FunctionName
                           << (ptype << DontCare << clone(tn) << TypeArgs)
                           << clone(id) << TypeArgs)
                       << args);

            auto lhs = _[Lhs];
            auto rhs = _[Rhs];

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
              // lambda to get the default argument, checking for nonlocal.
              auto def_arg = Call << apply()
                                  << (Args << (Expr << (*it / Default)));
              def_arg = nlrexpand(_, def_arg, true);

              // Add the default argument to the forwarding call.
              args << (Expr << def_arg);

              // Add a new function that calls the arity+1 function.
              seq
                << (Function << clone(ref) << clone(id) << clone(tp)
                             << clone(params) << clone(ty) << DontCare
                             << clone(pred) << (Block << clone(fwd)));

              // Add a parameter.
              auto param_id = *it / Ident;
              params << (Param << clone(param_id) << clone(*it / Type));

              // Replace the last argument with a reference to the parameter.
              args->pop_back();
              args << (Expr << (RefLet << clone(param_id)));
            }

            // The original function, with no default arguments.
            return seq
              << (Function << ref << id << tp << params << ty << DontCare
                           << pred << _(Block));
          },

        T(Param) << (T(Ident)[Ident] * T(Type)[Type] * T(DontCare)) >>
          [](Match& _) { return Param << _(Ident) << _(Type); },

        T(Param)[Param] << (T(Ident) * T(Type) * T(Call)) >>
          [](Match& _) {
            return err(
              _[Param],
              "can't put a default value before a non-defaulted value");
          },
      }};
  }
}

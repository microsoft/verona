// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"

namespace verona
{
  PassDef namearity()
  {
    return {
      dir::bottomup | dir::once,
      {
        T(Function)
            << (IsImplicit[Implicit] * Hand[Ref] * T(Ident)[Ident] *
                T(TypeParams)[TypeParams] * T(Params)[Params] * T(Type)[Type] *
                (T(LLVMFuncType) / T(DontCare))[LLVMFuncType] *
                T(TypePred)[TypePred] * (T(Block) / T(DontCare))[Block]) >>
          [](Match& _) {
            auto id = _(Ident);
            auto arity = _(Params)->size();
            auto name =
              std::string(id->location().view()) + "." + std::to_string(arity);

            if (_(Ref)->type() == Lhs)
              name += ".lhs";

            return Function << _(Implicit) << (Ident ^ name) << _(TypeParams)
                            << _(Params) << _(Type) << _(LLVMFuncType)
                            << _(TypePred) << _(Block);
          },

        (T(Call) / T(CallLHS))[Call]
            << ((T(FunctionName)
                 << ((TypeName / T(DontCare))[Lhs] * T(Ident)[Ident] *
                     T(TypeArgs)[TypeArgs])) *
                T(Args)[Args]) >>
          [](Match& _) {
            auto arity = _(Args)->size();
            auto name = std::string(_(Ident)->location().view()) + "." +
              std::to_string(arity);

            if (_(Call)->type() == CallLHS)
              name += ".ref";

            return Call << (FunctionName << _(Lhs) << (Ident ^ name)
                                         << _(TypeArgs))
                        << _(Args);
          },

        (T(Call) / T(CallLHS))[Call]
            << ((T(Selector) << (T(Ident)[Ident] * T(TypeArgs)[TypeArgs])) *
                T(Args)[Args]) >>
          [](Match& _) {
            auto arity = _(Args)->size();
            auto name = std::string(_(Ident)->location().view()) + "." +
              std::to_string(arity);

            if (_(Call)->type() == CallLHS)
              name += ".lhs";

            return Call << (Selector << (Ident ^ name) << _(TypeArgs))
                        << _(Args);
          },

        T(Call) << (T(New) * T(Args)[Args]) >>
          [](Match& _) {
            auto arity = _(Args)->size();
            auto name = std::string("new.") + std::to_string(arity);
            return Call << (FunctionName << DontCare << (Ident ^ name)
                                         << TypeArgs)
                        << _(Args);
          },

        T(CallLHS)[Call] << T(New) >> ([](Match& _) -> Node {
          if (!is_implicit(_(Call)))
            return err(_[Call], "can't assign to new");

          return NoChange;
        }),
      }};
  }
}

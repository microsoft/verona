// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"

namespace verona
{
  PassDef namearity()
  {
    return {
      dir::bottomup | dir::once,
      {
        T(Function)
            << ((T(Ref) / T(DontCare))[Ref] * Name[Id] *
                T(TypeParams)[TypeParams] * T(Params)[Params] * T(Type)[Type] *
                (T(LLVMFuncType) / T(DontCare))[LLVMFuncType] *
                T(TypePred)[TypePred] * (T(Block) / T(DontCare))[Block]) >>
          [](Match& _) {
            auto id = _(Id);
            auto arity = _(Params)->size();
            auto name =
              std::string(id->location().view()) + "." + std::to_string(arity);

            if (_(Ref)->type() == Ref)
              name += ".ref";

            return Function << (Ident ^ name) << _(TypeParams) << _(Params)
                            << _(Type) << _(LLVMFuncType) << _(TypePred)
                            << _(Block);
          },

        (T(Call) / T(CallLHS))[Call]
            << ((T(FunctionName)
                 << ((TypeName / T(DontCare))[Lhs] * Name[Id] *
                     T(TypeArgs)[TypeArgs])) *
                T(Args)[Args]) >>
          [](Match& _) {
            auto arity = _(Args)->size();
            auto name = std::string(_(Id)->location().view()) + "." +
              std::to_string(arity);

            if (_(Call)->type() == CallLHS)
              name += ".ref";

            return Call << (FunctionName << _(Lhs) << (Ident ^ name)
                                         << _(TypeArgs))
                        << _(Args);
          },

        (T(Call) / T(CallLHS))[Call]
            << ((T(Selector) << (Name[Id] * T(TypeArgs)[TypeArgs])) *
                T(Args)[Args]) >>
          [](Match& _) {
            auto arity = _(Args)->size();
            auto name = std::string(_(Id)->location().view()) + "." +
              std::to_string(arity);

            if (_(Call)->type() == CallLHS)
              name += ".ref";

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

        T(CallLHS)[Call] << T(New) >>
          [](Match& _) { return err(_[Call], "can't assign to new"); },
      }};
  }
}

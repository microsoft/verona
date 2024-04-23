// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../lookup.h"
#include "../wf.h"

namespace verona
{
  PassDef defaultargs()
  {
    return {
      "defaultargs",
      wfPassDefaultArgs,
      dir::bottomup | dir::once,
      {
        T(Function)
            << (IsImplicit[Implicit] * Hand[Ref] * T(Ident)[Ident] *
                T(TypeParams)[TypeParams] * T(Params)[Params] * T(Type)[Type] *
                T(LLVMFuncType, DontCare)[LLVMFuncType] *
                T(TypePred)[TypePred] * T(Block, DontCare)[Block]) >>
          [](Match& _) {
            Node seq = Seq;
            auto implicit = _(Implicit)->type();
            auto hand = _(Ref)->type();
            auto id = _(Ident);
            auto tp = _(TypeParams);
            auto params = _(Params);
            auto ty = _(Type);
            auto llvmty = _(LLVMFuncType);
            auto pred = _(TypePred);

            Node new_params = Params;
            Node args = Tuple;
            bool has_default = false;

            for (auto& param : *params)
            {
              auto param_id = param / Ident;
              auto param_type = param / Type;
              auto block = param / Default;
              args << (Expr << (RefLet << clone(param_id)));

              if (block == DontCare)
              {
                if (has_default)
                {
                  new_params << err(
                    param,
                    "Can't put a parameter with no default value after a "
                    "parameter with one");
                  // Do not process further to prevent duplicate errors.
                  break;
                }
              }
              else
              {
                has_default = true;
                auto def_arg = block->back();

                // Syntactically, the last statement in the block is the default
                // argument expression. WF doesn't enforce this.
                if (def_arg == Expr)
                  block->pop_back();
                else
                  def_arg = Expr << Unit;

                // Evaluate the default argument and call the arity+1 function.
                block << (Expr
                          << (Assign << (Expr << (Let << (Ident ^ param_id)))
                                     << def_arg))
                      << (Expr << Self << DoubleColon << selector(id)
                               << tuple_to_args(clone(args)));

                // Add a new function that calls the arity+1 function. Mark it
                // as explicit, so that errors when type checking the default
                // arguments are reported.
                seq
                  << (Function << implicit << hand << clone(id) << clone(tp)
                               << clone(new_params) << clone(ty)
                               << clone(llvmty) << clone(pred) << block);
              }

              // Add the parameter to the new parameter list.
              new_params << (Param << clone(param_id) << clone(param_type));
            }

            // The original function, with no default arguments.
            return seq
              << (Function << implicit << hand << id << tp << new_params << ty
                           << llvmty << pred << _(Block));
          },

        // Strip the default field values.
        T(FieldLet)
            << (IsImplicit[Implicit] * T(Ident)[Ident] * T(Type)[Type] * Any) >>
          [](Match& _) {
            return FieldLet << _(Implicit) << _(Ident) << _(Type);
          },

        T(FieldVar)
            << (IsImplicit[Implicit] * T(Ident)[Ident] * T(Type)[Type] * Any) >>
          [](Match& _) {
            return FieldVar << _(Implicit) << _(Ident) << _(Type);
          },
      }};
  }
}

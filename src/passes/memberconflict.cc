// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"

namespace verona
{
  std::pair<size_t, size_t> arity(Node& func)
  {
    auto params = func / Params;
    auto arity_hi = params->size();
    auto arity_lo = arity_hi;

    for (auto& param : *params)
    {
      if ((param / Default)->type() != DontCare)
        arity_lo--;
    }

    return {arity_lo, arity_hi};
  }

  PassDef memberconflict()
  {
    return {
      dir::topdown | dir::once,
      {
        (T(FieldLet) / T(FieldVar))[Op] << (T(Ident)[Ident]) >>
          ([](Match& _) -> Node {
            // Fields can conflict with other fields.
            auto field = _(Op);
            auto defs = field->scope()->lookdown(_(Ident)->location());

            for (auto& def : defs)
            {
              if (def->type().in({FieldLet, FieldVar}) && def->precedes(field))
                return err(field, "duplicate field name")
                  << (ErrorAst ^ (def / Ident));
            }

            return NoChange;
          }),

        T(Function)[Function]
            << (IsImplicit * Hand[Ref] * T(Ident)[Ident] * T(TypeParams) *
                T(Params)[Params] * T(Type) * (T(LLVMFuncType) / T(DontCare)) *
                T(TypePred) * (T(Block) / T(DontCare))[Block]) >>
          ([](Match& _) -> Node {
            // Functions can conflict with types, functions of the same arity
            // and handedness, and fields if the function is arity 1.
            auto func = _(Function);

            // Functions in classes must have implementations.
            if (
              (func->parent({Class, TypeTrait})->type() == Class) &&
              (_(Block)->type() == DontCare))
            {
              return err(
                func, "functions in classes must have implementations");
            }

            auto hand = _(Ref)->type();
            auto [arity_lo, arity_hi] = arity(func);
            auto defs = func->scope()->lookdown(_(Ident)->location());

            for (auto& def : defs)
            {
              if (
                (def->type() == Function) && ((def / Ref)->type() == hand) &&
                def->precedes(func))
              {
                auto [def_arity_lo, def_arity_hi] = arity(def);

                if ((def_arity_hi >= arity_lo) && (def_arity_lo <= arity_hi))
                  return err(
                           func,
                           "this function has the same name, arity, and "
                           "handedness as "
                           "another function")
                    << (ErrorAst ^ (def / Ident));
              }
              else if (
                (def->type() == FieldLet) && (hand == Rhs) && (1 >= arity_lo) &&
                (1 <= arity_hi))
              {
                return err(func, "this function has the same arity as a field")
                  << (ErrorAst ^ (def / Ident));
              }
              else if (
                (def->type() == FieldVar) && (1 >= arity_lo) && (1 <= arity_hi))
              {
                return err(func, "this function has the same arity as a field")
                  << (ErrorAst ^ (def / Ident));
              }
            }

            return NoChange;
          }),
      }};
  }
}

// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"

namespace verona
{
  PassDef autocreate()
  {
    return {
      dir::topdown | dir::once,
      {
        In(Class) * T(ClassBody)[ClassBody] >> ([](Match& _) -> Node {
          // If we already have a create function, do nothing.
          auto class_body = _(ClassBody);
          Node new_params = Params;
          Node new_args = Args;

          for (auto& node : *class_body)
          {
            if (node->type().in({FieldLet, FieldVar}))
            {
              auto id = node / Ident;
              auto ty = node / Type;
              auto def_arg = node / Default;

              // Add each field in order to the call to `new` and the create
              // function parameters.
              new_args << (Expr << (RefLet << clone(id)));
              new_params
                << ((Param ^ def_arg) << clone(id) << clone(ty) << def_arg);
            }
          }

          // Create the `new` function.
          // TODO: return Self & K?
          auto body = ClassBody
            << *_[ClassBody]
            << (Function << DontCare << (Ident ^ new_) << TypeParams
                         << new_params << typevar(_) << DontCare << typepred()
                         << (Block << (Expr << unit())));

          if (class_body->parent()->lookdown(create).empty())
          {
            // Create the `create` function.
            body
              << (Function << DontCare << (Ident ^ create) << TypeParams
                           << clone(new_params) << typevar(_) << DontCare
                           << typepred()
                           << (Block << (Expr << (Call << New << new_args))));
          }

          return body;
        }),

        // Strip the default field values.
        T(FieldLet) << (T(Ident)[Id] * T(Type)[Type] * Any) >>
          [](Match& _) { return FieldLet << _(Id) << _(Type); },

        T(FieldVar) << (T(Ident)[Id] * T(Type)[Type] * Any) >>
          [](Match& _) { return FieldVar << _(Id) << _(Type); },
      }};
  }
}

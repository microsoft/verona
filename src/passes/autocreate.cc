// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../lookup.h"

namespace verona
{
  PassDef autocreate()
  {
    return {
      dir::topdown | dir::once,
      {
        In(Class) * T(ClassBody)[ClassBody] >> ([](Match& _) -> Node {
          auto class_body = _(ClassBody);
          Node new_params = Params;
          Node new_args = Tuple;

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

          // Create the `new` function, with default arguments set to the field
          // initializers. Mark `new` as explicit, so that errors when type
          // checking `new` are reported.
          auto body = ClassBody
            << *_[ClassBody]
            << (Function << Explicit << Rhs << (Ident ^ l_new) << TypeParams
                         << new_params << typevar(_) << DontCare << typepred()
                         << (Block << (Expr << unit())));

          // If we already have a create function, don't emit one.
          if (class_body->parent()->lookdown(l_create).empty())
          {
            // Create the `create` function.
            auto fq_new = append_fq(local_fq(_(ClassBody)), selector(l_new));

            body
              << (Function << Implicit << Rhs << (Ident ^ l_create)
                           << TypeParams << clone(new_params) << typevar(_)
                           << DontCare << typepred()
                           << (Block << (Expr << call(fq_new, new_args))));
          }

          return body;
        }),

        // Strip the default field values.
        T(FieldLet) << (T(Ident)[Ident] * T(Type)[Type] * Any) >>
          [](Match& _) { return FieldLet << _(Ident) << _(Type); },

        T(FieldVar) << (T(Ident)[Ident] * T(Type)[Type] * Any) >>
          [](Match& _) { return FieldVar << _(Ident) << _(Type); },
      }};
  }
}

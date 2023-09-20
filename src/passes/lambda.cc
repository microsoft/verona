// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../lookup.h"

namespace verona
{
  PassDef lambda()
  {
    auto freevars = std::make_shared<std::vector<std::set<Location>>>();

    PassDef lambda = {
      dir::bottomup | dir::once,
      {
        T(RefLet) << T(Ident)[Ident] >> ([freevars](Match& _) -> Node {
          if (!freevars->empty())
          {
            // If we don't have a definition within the scope of the lambda,
            // then it's a free variable.
            auto id = _(Ident);

            if (id->lookup(id->parent(Lambda)).empty())
              freevars->back().insert(id->location());
          }

          return NoChange;
        }),

        T(Lambda)[Lambda]
            << (T(TypeParams)[TypeParams] * T(Params)[Params] * T(Type)[Type] *
                T(TypePred)[TypePred] * T(Block)[Block]) >>
          [freevars](Match& _) {
            // Create the anonymous type.
            Node class_body = ClassBody;
            auto class_id = _.fresh(l_lambda);
            auto classdef = Class << (Ident ^ class_id) << TypeParams
                                  << (Inherit << DontCare) << typepred()
                                  << class_body;

            // Build an FQType for the anonymous type.
            auto fq = append_fq(
              local_fq(_(Lambda)->parent({Class, Trait})),
              TypeClassName << (Ident ^ class_id) << TypeArgs);

            // The create function will capture the free variables.
            auto fq_new = append_fq(fq, selector(l_new));
            Node new_args = Tuple;

            auto fq_create = append_fq(fq, selector(l_create));
            Node create_params = Params;
            Node create_args = Tuple;

            Node apply_body = Block;
            auto self_id = _.fresh(l_self);
            auto& fv = freevars->back();

            std::for_each(fv.begin(), fv.end(), [&](auto& fv_id) {
              // Add a field for the free variable to the anonymous type.
              auto type_id = _.fresh(l_typevar);
              class_body
                << (FieldLet << Explicit << (Ident ^ fv_id) << typevar(type_id)
                             << DontCare);

              // Add a parameter to the create function to capture the free
              // variable as a field.
              create_params
                << (Param << (Ident ^ fv_id) << typevar(type_id) << DontCare);
              new_args << (Expr << (RefLet << (Ident ^ fv_id)));

              // Add an argument to the create call. Don't load the free
              // variable, even if it was a `var`.
              create_args << (Expr << (RefLet << (Ident ^ fv_id)));

              // At the start of the lambda body, assign the field to a
              // local variable with the same name as the free variable.
              apply_body
                << (Expr
                    << (Assign
                        << (Expr
                            << (TypeAssert << (Expr << (Let << (Ident ^ fv_id)))
                                           << typevar(type_id)))
                        << (Expr << call(
                              selector(fv_id), RefLet << (Ident ^ self_id)))));
            });

            // The apply function is the original lambda. Prepend a
            // `self`-like parameter with a fresh name to the lambda
            // parameters.
            // TODO: capability for Self
            auto apply_func = Function
              << LambdaFunc << Rhs << (Ident ^ l_apply) << _(TypeParams)
              << (Params << (Param << (Ident ^ self_id) << (Type << Self)
                                   << DontCare)
                         << *_[Params])
              << _(Type) << DontCare << _(TypePred)
              << (apply_body << *_[Block]);

            // Add the create and apply functions to the anonymous type.
            auto create_func = Function
              << Implicit << Rhs << (Ident ^ l_create) << TypeParams
              << create_params << typevar(_) << DontCare << typepred()
              << (Block << (Expr << call(fq_new, new_args)));

            class_body << create_func << apply_func;
            freevars->pop_back();

            Token target =
              _(Lambda)->parent()->in({FieldLet, FieldVar, Param}) ? ClassBody :
                                                                     Block;

            return Seq << (Lift << target << classdef)
                       << call(fq_create, create_args);
          },
      }};

    lambda.pre(Lambda, [freevars](Node) {
      freevars->push_back({});
      return 0;
    });

    return lambda;
  }
}

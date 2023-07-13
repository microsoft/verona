// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"

namespace verona
{
  auto make_conditional(Match& _)
  {
    // Pack all of the branches into a single conditional and unpack them
    // in the follow-on rules.
    auto lambda = _(Lhs);
    auto params = lambda / Params;
    Node cond = Expr;
    Node block = Block;
    Node args;

    if (params->empty())
    {
      // This is a boolean conditional.
      cond << _[Expr];
      args = Unit;
    }
    else
    {
      // This is a TypeTest conditional.
      auto id = _.fresh();
      Node lhs;
      Node type;

      if (params->size() == 1)
      {
        // This is a single parameter.
        auto lhs_id = _.fresh();
        lhs = Expr << (Let << (Ident ^ lhs_id));
        type = clone(params->front() / Type);
        args = Ident ^ lhs_id;
      }
      else
      {
        // This is multiple parameters. We need to build a TypeTuple for the
        // Cast and a Tuple both for destructuring the cast value and for the
        // arguments to be passed to the lambda on success.
        args = Tuple;
        lhs = Tuple;

        Node typetuple = TypeTuple;
        type = Type << typetuple;

        for (auto& param : *params)
        {
          auto lhs_id = _.fresh();
          args << (Expr << (Ident ^ lhs_id));
          lhs << (Expr << (Let << (Ident ^ lhs_id)));
          typetuple << clone(param / Type / Type);
        }
      }

      cond
        << (TypeTest << (Expr
                         << (Assign << (Expr << (Let << (Ident ^ id)))
                                    << (Expr << _[Expr])))
                     << clone(type));

      block
        << (Expr
            << (Assign << (Expr << lhs)
                       << (Expr << (Cast << (Expr << (Ident ^ id)) << type))));
    }

    return Conditional << cond << (block << (Expr << lambda << args))
                       << (Block << (Expr << (Conditional << _[Rhs])));
  }

  PassDef conditionals()
  {
    return {
      // Conditionals are right-associative.
      In(Expr) * T(If) * (!T(Lambda) * (!T(Lambda))++)[Expr] * T(Lambda)[Lhs] *
          ((T(Else) * T(If) * (!T(Lambda) * (!T(Lambda))++) * T(Lambda))++ *
           ~(T(Else) * T(Lambda)))[Rhs] >>
        [](Match& _) { return make_conditional(_); },

      T(Conditional)
          << ((T(Else) * T(If) * (!T(Lambda) * (!T(Lambda))++)[Expr] *
               T(Lambda)[Lhs]) *
              Any++[Rhs]) >>
        [](Match& _) { return make_conditional(_); },

      T(Conditional) << (T(Else) * T(Lambda)[Rhs] * End) >>
        [](Match& _) { return Seq << _(Rhs) << Unit; },

      T(Conditional) << End >> ([](Match&) -> Node { return Unit; }),

      T(If)[If] >>
        [](Match& _) {
          return err(_[If], "`if` must be followed by a condition and braces");
        },

      T(Else)[Else] >>
        [](Match& _) {
          return err(
            _[Else],
            "`else` must follow an `if` and be followed by an `if` or braces");
        },
    };
  }
}
